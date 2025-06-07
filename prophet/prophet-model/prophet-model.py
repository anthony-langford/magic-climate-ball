import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Prophet installation check
try:
    from prophet import Prophet

    print("✓ Prophet is available")
except ImportError:
    print("⚠ Prophet not installed. Install with: pip install prophet")
    print(
        "Note: Prophet requires additional dependencies and may take a few minutes to install"
    )
    exit()


class ProphetTemperatureForecaster:
    """Prophet-based temperature forecasting - much lighter than SARIMA"""

    def __init__(self, data_path="temagami_features.csv"):
        """Load the temperature data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        self.prophet_data = pd.DataFrame(
            {"ds": self.df.index, "y": self.df["t_mean"]}
        ).dropna()

        print(f"Loaded temperature data: {len(self.prophet_data)} observations")
        print(
            f"Date range: {self.prophet_data['ds'].min().date()} to {self.prophet_data['ds'].max().date()}"
        )

    def prepare_regressors(self):
        """Add external regressors from our engineered features"""
        # Add useful features as regressors
        prophet_data = self.prophet_data.copy()

        useful_features = [
            "lag_1",
            "lag_7",
            "lag_365",
            "roll_mean_7",
            "roll_mean_30",
            "anomaly_30",
            "sin_doy",
            "cos_doy",
        ]

        available_features = []
        for feature in useful_features:
            if feature in self.df.columns:
                # Align data
                feature_data = self.df[feature].reindex(prophet_data["ds"])
                if not feature_data.isna().all():
                    prophet_data[feature] = feature_data
                    available_features.append(feature)

        print(f"Added {len(available_features)} regressors: {available_features}")
        self.prophet_data_with_regressors = prophet_data
        self.available_regressors = available_features

        return prophet_data

    def create_prophet_models(self):
        """Create different Prophet model configurations"""
        models = {}

        # 1. Basic Prophet (no regressors)
        models["basic"] = {
            "name": "Basic Prophet",
            "config": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,  # Less relevant for temperature
                "daily_seasonality": False,
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 0.05,  # Conservative changepoints
                "n_changepoints": 25,
            },
            "regressors": [],
        }

        # 2. Prophet with seasonality tuning
        models["tuned_seasonality"] = {
            "name": "Tuned Seasonality",
            "config": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "seasonality_mode": "multiplicative",  # Temperature seasonality can be multiplicative
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 15.0,  # Stronger seasonality
                "n_changepoints": 50,
            },
            "regressors": [],
        }

        # 3. Prophet with lag features
        models["with_lags"] = {
            "name": "With Lag Features",
            "config": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 0.05,
            },
            "regressors": (
                ["lag_1", "lag_7"] if hasattr(self, "available_regressors") else []
            ),
        }

        # 4. Full model with all available regressors
        if hasattr(self, "available_regressors"):
            models["full"] = {
                "name": "Full Model",
                "config": {
                    "yearly_seasonality": True,
                    "weekly_seasonality": False,
                    "daily_seasonality": False,
                    "seasonality_mode": "additive",
                    "changepoint_prior_scale": 0.1,
                    "seasonality_prior_scale": 10.0,
                },
                "regressors": self.available_regressors[
                    :5
                ],  # Limit to avoid overfitting
            }

        self.model_configs = models
        return models

    def fit_prophet_model(self, model_config, train_data):
        """Fit a single Prophet model"""
        try:
            # Create Prophet model
            model = Prophet(**model_config["config"])

            # Add regressors
            for regressor in model_config["regressors"]:
                if regressor in train_data.columns:
                    model.add_regressor(regressor)

            # Fit model (Prophet is generally fast and stable)
            # Remove verbose parameter for newer Prophet versions
            model.fit(train_data[["ds", "y"] + model_config["regressors"]])

            return model

        except Exception as e:
            print(f"Error fitting {model_config['name']}: {e}")
            return None

    def walk_forward_validation(self, test_years=2, forecast_horizon=30):
        """Walk-forward validation for Prophet models"""
        print(f"\n" + "=" * 50)
        print("PROPHET WALK-FORWARD VALIDATION")
        print("=" * 50)

        # Prepare data with regressors
        if not hasattr(self, "prophet_data_with_regressors"):
            self.prepare_regressors()

        # Create model configurations
        self.create_prophet_models()

        # Split data
        test_start_date = self.prophet_data["ds"].max() - pd.Timedelta(
            days=365 * test_years
        )
        train_mask = self.prophet_data_with_regressors["ds"] <= test_start_date
        test_mask = self.prophet_data_with_regressors["ds"] > test_start_date

        print(f"Training: {train_mask.sum()} observations")
        print(f"Testing: {test_mask.sum()} observations")
        print(f"Split date: {test_start_date.date()}")

        # Test forecast origins (every 2 weeks to keep it manageable)
        test_data = self.prophet_data_with_regressors[test_mask].copy()
        forecast_origins = test_data["ds"][::14].values  # Every 14 days

        print(f"Testing {len(forecast_origins)} forecast origins...")

        all_results = []

        # Test each model configuration
        for model_name, model_config in self.model_configs.items():
            print(f"\nTesting {model_config['name']}...")

            model_results = []

            for i, origin_date in enumerate(forecast_origins):
                if i % 5 == 0:  # Progress update every 5 origins
                    print(
                        f"  Origin {i+1}/{len(forecast_origins)}: {pd.to_datetime(origin_date).date()}"
                    )

                try:
                    # Expanding window training data
                    current_train_mask = (
                        self.prophet_data_with_regressors["ds"] < origin_date
                    )
                    current_train = self.prophet_data_with_regressors[
                        current_train_mask
                    ].copy()

                    if len(current_train) < 365:  # Need at least 1 year
                        continue

                    # Fit model
                    model = self.fit_prophet_model(model_config, current_train)
                    if model is None:
                        continue

                    # Create future dataframe
                    future_dates = pd.date_range(
                        start=origin_date, periods=forecast_horizon + 1, freq="D"
                    )[
                        1:
                    ]  # Exclude origin date

                    future_df = pd.DataFrame({"ds": future_dates})

                    # Add regressor values for future dates
                    for regressor in model_config["regressors"]:
                        if regressor in self.prophet_data_with_regressors.columns:
                            # Use actual values if available, otherwise forward fill
                            regressor_values = []
                            for future_date in future_dates:
                                if (
                                    future_date
                                    in self.prophet_data_with_regressors["ds"].values
                                ):
                                    idx = (
                                        self.prophet_data_with_regressors["ds"]
                                        == future_date
                                    )
                                    value = self.prophet_data_with_regressors.loc[
                                        idx, regressor
                                    ].iloc[0]
                                else:
                                    # Forward fill from last known value
                                    last_idx = (
                                        self.prophet_data_with_regressors["ds"]
                                        < future_date
                                    )
                                    if last_idx.any():
                                        value = self.prophet_data_with_regressors.loc[
                                            last_idx, regressor
                                        ].iloc[-1]
                                    else:
                                        value = 0  # Fallback
                                regressor_values.append(value)

                            future_df[regressor] = regressor_values

                    # Generate forecast
                    forecast = model.predict(future_df)

                    # Compare with actual values
                    for j, future_date in enumerate(future_dates):
                        if future_date in test_data["ds"].values:
                            actual_idx = test_data["ds"] == future_date
                            actual = test_data.loc[actual_idx, "y"].iloc[0]
                            predicted = forecast.loc[j, "yhat"]

                            model_results.append(
                                {
                                    "model": model_name,
                                    "origin_date": origin_date,
                                    "forecast_date": future_date,
                                    "horizon": j + 1,
                                    "actual": actual,
                                    "predicted": predicted,
                                    "error": abs(actual - predicted),
                                    "lower": forecast.loc[j, "yhat_lower"],
                                    "upper": forecast.loc[j, "yhat_upper"],
                                }
                            )

                except Exception as e:
                    print(
                        f"    Error at origin {pd.to_datetime(origin_date).date()}: {str(e)[:50]}..."
                    )
                    continue

            all_results.extend(model_results)
            print(
                f"  Collected {len(model_results)} forecasts for {model_config['name']}"
            )

        self.validation_results = pd.DataFrame(all_results)
        print(
            f"\nValidation complete: {len(self.validation_results)} total forecast-actual pairs"
        )

        return self.validation_results

    def evaluate_prophet_performance(self):
        """Evaluate Prophet model performance"""
        if not hasattr(self, "validation_results") or len(self.validation_results) == 0:
            print("No validation results available")
            return None

        print(f"\n" + "=" * 60)
        print("PROPHET MODEL PERFORMANCE")
        print("=" * 60)

        # Performance by model and horizon
        performance = (
            self.validation_results.groupby(["model", "horizon"])["error"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        performance.columns = ["MAE", "Std", "Count"]

        # Show performance for key horizons
        key_horizons = [1, 7, 14, 21, 30]

        for model_name in self.validation_results["model"].unique():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            print(f"{'Horizon':<8} {'MAE (°C)':<10} {'Count':<8}")
            print("-" * 40)

            for horizon in key_horizons:
                if (model_name, horizon) in performance.index:
                    row = performance.loc[(model_name, horizon)]
                    print(f"{horizon:<8} {row['MAE']:<10.2f} {int(row['Count']):<8}")

        return performance

    def compare_prophet_models(self):
        """Compare different Prophet configurations"""
        if not hasattr(self, "validation_results") or len(self.validation_results) == 0:
            print("No validation results available - all models failed to fit")
            return None

        # Average performance across all horizons
        model_comparison = (
            self.validation_results.groupby("model")["error"]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        model_comparison.columns = ["Avg_MAE", "Std_MAE", "N_Forecasts"]
        model_comparison = model_comparison.sort_values("Avg_MAE")

        print(f"\nPROPHET MODEL COMPARISON (Average across all horizons):")
        print("-" * 60)
        print(f"{'Model':<20} {'Avg MAE':<10} {'Std MAE':<10} {'N Forecasts':<12}")
        print("-" * 60)

        for model_name, row in model_comparison.iterrows():
            config_name = self.model_configs[model_name]["name"]
            print(
                f"{config_name:<20} {row['Avg_MAE']:<10.2f} {row['Std_MAE']:<10.2f} {int(row['N_Forecasts']):<12}"
            )

        return model_comparison

    def compare_with_baselines(self, baseline_path="baseline_results.csv"):
        """Compare best Prophet model with baselines"""
        try:
            baseline_df = pd.read_csv(baseline_path)
        except FileNotFoundError:
            print("Baseline results not found")
            return

        if not hasattr(self, "validation_results"):
            print("No Prophet results to compare")
            return

        # Get best Prophet model performance by horizon
        best_model = self.validation_results.groupby("model")["error"].mean().idxmin()
        prophet_performance = (
            self.validation_results[self.validation_results["model"] == best_model]
            .groupby("horizon")["error"]
            .mean()
        )

        print(f"\nBEST PROPHET MODEL vs BASELINES:")
        print(f"Best model: {self.model_configs[best_model]['name']}")
        print("-" * 70)
        print(
            f"{'Horizon':<8} {'Prophet':<10} {'Climatology':<12} {'Best Base':<12} {'Improvement':<12}"
        )
        print("-" * 70)

        for horizon in [1, 3, 7, 14, 21, 30]:
            baseline_row = baseline_df[baseline_df["Horizon"] == horizon]

            if len(baseline_row) > 0 and horizon in prophet_performance.index:
                prophet_mae = prophet_performance[horizon]
                clim_mae = baseline_row["Climatology_MAE"].iloc[0]
                best_baseline = min(
                    baseline_row["Climatology_MAE"].iloc[0],
                    baseline_row["Seasonal_Naive_MAE"].iloc[0],
                    baseline_row["Persistence_MAE"].iloc[0],
                )

                improvement = ((best_baseline - prophet_mae) / best_baseline) * 100

                print(
                    f"{horizon:<8} {prophet_mae:<10.2f} {clim_mae:<12.2f} {best_baseline:<12.2f} {improvement:+8.1f}%"
                )

    def plot_forecast_example(self, model_name="basic", days_to_show=90):
        """Plot an example Prophet forecast"""
        if not hasattr(self, "model_configs"):
            self.create_prophet_models()

        # Use recent data for example
        recent_data = self.prophet_data_with_regressors.tail(days_to_show + 30)
        train_data = recent_data.iloc[:-30]

        # Fit model
        model_config = self.model_configs[model_name]
        model = self.fit_prophet_model(model_config, train_data)

        if model is None:
            print("Could not fit model for plotting")
            return

        # Create future dataframe
        future = model.make_future_dataframe(periods=30)

        # Add regressors for future (simplified - just forward fill)
        for regressor in model_config["regressors"]:
            if regressor in train_data.columns:
                last_value = train_data[regressor].iloc[-1]
                future[regressor] = (
                    train_data[regressor].reindex(future.index).fillna(last_value)
                )

        # Generate forecast
        forecast = model.predict(future)

        # Plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot training data
        ax.plot(
            train_data["ds"], train_data["y"], "b-", label="Training Data", linewidth=1
        )

        # Plot actual test data if available
        test_data = recent_data.iloc[-30:]
        ax.plot(test_data["ds"], test_data["y"], "g-", label="Actual", linewidth=2)

        # Plot forecast
        forecast_test = forecast.iloc[-30:]
        ax.plot(
            forecast_test["ds"],
            forecast_test["yhat"],
            "r--",
            label="Prophet Forecast",
            linewidth=2,
        )

        # Plot uncertainty intervals
        ax.fill_between(
            forecast_test["ds"],
            forecast_test["yhat_lower"],
            forecast_test["yhat_upper"],
            alpha=0.3,
            color="red",
            label="Uncertainty",
        )

        # Formatting
        ax.axvline(
            x=train_data["ds"].iloc[-1],
            color="black",
            linestyle=":",
            alpha=0.7,
            label="Forecast Start",
        )
        ax.set_title(f'Prophet Temperature Forecast Example ({model_config["name"]})')
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("prophet_forecast_example.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Calculate MAE for this example
        mae = mean_absolute_error(test_data["y"], forecast_test["yhat"])
        print(f"Example forecast MAE: {mae:.2f}°C")


def main():
    """Main Prophet pipeline - much lighter than SARIMA"""
    print("Starting Prophet temperature forecasting pipeline...")
    print("This should run smoothly without system stress!")

    # Initialize forecaster
    forecaster = ProphetTemperatureForecaster("temagami_features.csv")

    # Prepare features
    forecaster.prepare_regressors()

    # Run validation (this is much faster than SARIMA)
    print("\nRunning walk-forward validation...")
    validation_results = forecaster.walk_forward_validation(
        test_years=2, forecast_horizon=30
    )

    # Evaluate performance
    performance = forecaster.evaluate_prophet_performance()

    # Compare models (only if we have results)
    if len(validation_results) > 0:
        model_comparison = forecaster.compare_prophet_models()

        # Compare with baselines
        forecaster.compare_with_baselines()

        # Create example plot
        print("\nCreating forecast example...")
        forecaster.plot_forecast_example()

        print("\nProphet analysis complete! Much gentler on your Mac 😊")
    else:
        print(
            "\n⚠ All Prophet models failed to fit - this is likely a version compatibility issue"
        )
        print("Let's try a simpler approach...")

        # Try a single basic forecast as fallback
        try:
            simple_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )

            # Use just recent data to test
            recent_data = forecaster.prophet_data.tail(1000)
            train_data = recent_data.iloc[:-30]
            test_data = recent_data.iloc[-30:]

            simple_model.fit(train_data)

            future = simple_model.make_future_dataframe(periods=30)
            forecast = simple_model.predict(future)

            # Calculate MAE for the test period
            test_forecast = forecast.tail(30)["yhat"].values
            mae = mean_absolute_error(test_data["y"].values, test_forecast)

            print(f"✓ Simple Prophet model worked! Test MAE: {mae:.2f}°C")
            print(
                "This suggests Prophet can work, but there might be issues with the validation setup"
            )

        except Exception as e:
            print(f"✗ Even simple Prophet failed: {e}")
            print("This suggests a deeper Prophet installation issue")

        model_comparison = None

    return forecaster, validation_results, performance


if __name__ == "__main__":
    forecaster, results, performance = main()
