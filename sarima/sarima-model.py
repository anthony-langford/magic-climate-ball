import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import itertools

warnings.filterwarnings("ignore")


class SARIMAForecaster:
    """SARIMA model for temperature forecasting with walk-forward validation"""

    def __init__(self, data_path="temagami_features.csv"):
        """Load the feature-engineered data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.temp_series = self.df["t_mean"].copy()
        print(f"Loaded temperature series: {len(self.temp_series)} observations")
        print(
            f"Date range: {self.temp_series.index.min().date()} to {self.temp_series.index.max().date()}"
        )

    def check_stationarity(self, series, title="Temperature Series"):
        """Check if series is stationary using Augmented Dickey-Fuller test"""
        print(f"\nStationarity Test for {title}")
        print("-" * 40)

        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())

        print(f"ADF Statistic: {adf_result[0]:.6f}")
        print(f"p-value: {adf_result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in adf_result[4].items():
            print(f"\t{key}: {value:.3f}")

        if adf_result[1] <= 0.05:
            print("✓ Series is stationary (reject null hypothesis)")
            return True
        else:
            print("✗ Series is non-stationary (fail to reject null hypothesis)")
            return False

    def difference_series(self, series, seasonal_periods=365):
        """Apply regular and seasonal differencing"""
        print(f"\nApplying differencing...")

        # First difference
        diff1 = series.diff().dropna()
        is_stationary = self.check_stationarity(diff1, "First Differenced")

        if is_stationary:
            print("First differencing achieved stationarity")
            return diff1, (1, 0)

        # Seasonal difference
        seasonal_diff = series.diff(seasonal_periods).dropna()
        is_seasonal_stationary = self.check_stationarity(
            seasonal_diff, "Seasonal Differenced"
        )

        if is_seasonal_stationary:
            print("Seasonal differencing achieved stationarity")
            return seasonal_diff, (0, 1)

        # Both regular and seasonal differencing
        both_diff = series.diff().diff(seasonal_periods).dropna()
        is_both_stationary = self.check_stationarity(both_diff, "Both Differenced")

        if is_both_stationary:
            print("Both regular and seasonal differencing achieved stationarity")
            return both_diff, (1, 1)

        print("Warning: Could not achieve stationarity with standard differencing")
        return diff1, (1, 0)

    def plot_diagnostics(self, series, title="Temperature Series"):
        """Plot ACF and PACF for model identification"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series plot
        series.plot(ax=axes[0, 0], title=f"{title} - Time Series")
        axes[0, 0].set_ylabel("Temperature (°C)")

        # ACF plot
        plot_acf(series.dropna(), ax=axes[0, 1], lags=50, title=f"{title} - ACF")

        # PACF plot
        plot_pacf(series.dropna(), ax=axes[1, 0], lags=50, title=f"{title} - PACF")

        # Distribution
        series.hist(bins=50, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title(f"{title} - Distribution")
        axes[1, 1].set_xlabel("Temperature (°C)")

        plt.tight_layout()
        plt.show()

    def grid_search_sarima(self, train_data, max_order=2, seasonal_periods=365):
        """Grid search for best SARIMA parameters"""
        print(f"\nGrid searching SARIMA parameters...")
        print("This may take several minutes...")

        # Define parameter ranges
        p_values = range(0, max_order + 1)
        d_values = [0, 1]  # Based on stationarity tests
        q_values = range(0, max_order + 1)

        # Seasonal parameters (keep small for computational efficiency)
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]

        best_aic = np.inf
        best_params = None
        best_seasonal_params = None
        results = []

        total_combinations = (
            len(p_values)
            * len(d_values)
            * len(q_values)
            * len(P_values)
            * len(D_values)
            * len(Q_values)
        )
        current_combination = 0

        for p, d, q in itertools.product(p_values, d_values, q_values):
            for P, D, Q in itertools.product(P_values, D_values, Q_values):
                current_combination += 1

                if current_combination % 10 == 0:
                    print(
                        f"Progress: {current_combination}/{total_combinations} combinations tested"
                    )

                try:
                    # Fit SARIMA model
                    model = SARIMAX(
                        train_data,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_periods),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )

                    fitted_model = model.fit(disp=False, maxiter=100)

                    # Store results
                    results.append(
                        {
                            "order": (p, d, q),
                            "seasonal_order": (P, D, Q, seasonal_periods),
                            "aic": fitted_model.aic,
                            "bic": fitted_model.bic,
                            "converged": fitted_model.mle_retvals["converged"],
                        }
                    )

                    # Update best model
                    if (
                        fitted_model.aic < best_aic
                        and fitted_model.mle_retvals["converged"]
                    ):
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_seasonal_params = (P, D, Q, seasonal_periods)

                except Exception as e:
                    # Skip problematic parameter combinations
                    continue

        print(f"\nGrid search completed!")
        print(f"Best parameters: SARIMA{best_params} x {best_seasonal_params}")
        print(f"Best AIC: {best_aic:.2f}")

        # Show top 5 models
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df[results_df["converged"] == True]
            top_models = results_df.nsmallest(5, "aic")
            print(f"\nTop 5 models by AIC:")
            for idx, row in top_models.iterrows():
                print(
                    f"SARIMA{row['order']} x {row['seasonal_order']}: AIC={row['aic']:.2f}"
                )

        self.best_params = best_params
        self.best_seasonal_params = best_seasonal_params
        self.grid_search_results = results_df if len(results_df) > 0 else None

        return best_params, best_seasonal_params

    def fit_sarima(self, train_data, order=None, seasonal_order=None):
        """Fit SARIMA model with given or best parameters"""
        if order is None:
            order = getattr(self, "best_params", (1, 1, 1))
        if seasonal_order is None:
            seasonal_order = getattr(self, "best_seasonal_params", (1, 1, 1, 365))

        print(f"\nFitting SARIMA{order} x {seasonal_order}")

        try:
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted_model = model.fit(disp=False, maxiter=200)

            print(f"Model fitted successfully!")
            print(f"AIC: {fitted_model.aic:.2f}")
            print(f"BIC: {fitted_model.bic:.2f}")
            print(f"Log-likelihood: {fitted_model.llf:.2f}")

            self.fitted_model = fitted_model
            return fitted_model

        except Exception as e:
            print(f"Error fitting SARIMA model: {e}")
            return None

    def walk_forward_validation(self, test_years=3, forecast_horizon=30):
        """Walk-forward validation for SARIMA model"""
        print(f"\n" + "=" * 60)
        print("SARIMA WALK-FORWARD VALIDATION")
        print("=" * 60)

        # Split data
        test_start_year = self.temp_series.index.max().year - test_years + 1
        train_data = self.temp_series[self.temp_series.index.year < test_start_year]
        test_data = self.temp_series[self.temp_series.index.year >= test_start_year]

        print(
            f"Training: {train_data.index.min().date()} to {train_data.index.max().date()}"
        )
        print(
            f"Testing: {test_data.index.min().date()} to {test_data.index.max().date()}"
        )

        # Find best parameters on training data
        if not hasattr(self, "best_params"):
            print("Finding optimal parameters...")
            self.grid_search_sarima(train_data, max_order=2)

        # Walk-forward validation
        results = []
        forecast_dates = []

        # Start with initial training window
        current_train = train_data.copy()

        for test_date in test_data.index[::7]:  # Test every 7 days to speed up
            print(f"Forecasting from {test_date.date()}...")

            try:
                # Fit model on current training data
                model = SARIMAX(
                    current_train,
                    order=self.best_params,
                    seasonal_order=self.best_seasonal_params,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                fitted_model = model.fit(disp=False, maxiter=100)

                # Generate forecasts
                forecasts = fitted_model.forecast(steps=forecast_horizon)
                forecast_index = pd.date_range(
                    start=current_train.index[-1] + timedelta(days=1),
                    periods=forecast_horizon,
                    freq="D",
                )

                # Collect actual vs predicted for available dates
                for i, forecast_date in enumerate(forecast_index):
                    if forecast_date in test_data.index:
                        actual = test_data.loc[forecast_date]
                        predicted = (
                            forecasts.iloc[i]
                            if isinstance(forecasts, pd.Series)
                            else forecasts[i]
                        )

                        results.append(
                            {
                                "forecast_origin": test_date,
                                "forecast_date": forecast_date,
                                "horizon": i + 1,
                                "actual": actual,
                                "predicted": predicted,
                                "error": abs(actual - predicted),
                            }
                        )

                # Update training data (expanding window)
                if test_date in test_data.index:
                    current_train = pd.concat(
                        [current_train, test_data.loc[test_date:test_date]]
                    )

            except Exception as e:
                print(f"Error forecasting from {test_date.date()}: {e}")
                continue

        self.validation_results = pd.DataFrame(results)
        return self.validation_results

    def evaluate_sarima_performance(self):
        """Evaluate SARIMA performance by horizon"""
        if not hasattr(self, "validation_results"):
            print("Run walk_forward_validation first")
            return None

        # Calculate metrics by horizon
        horizon_metrics = (
            self.validation_results.groupby("horizon")
            .agg(
                {
                    "error": ["mean", "std", "count"],
                    "actual": "mean",
                    "predicted": "mean",
                }
            )
            .round(3)
        )

        horizon_metrics.columns = [
            "MAE",
            "STD_Error",
            "N_Forecasts",
            "Mean_Actual",
            "Mean_Predicted",
        ]

        print(f"\nSARIMA Performance by Horizon:")
        print("-" * 50)
        print(
            f"{'Horizon':<8} {'MAE':<8} {'N_Forecasts':<12} {'Actual':<8} {'Predicted':<10}"
        )
        print("-" * 50)

        for horizon in sorted(horizon_metrics.index):
            if horizon <= 30:  # Show up to 30 days
                row = horizon_metrics.loc[horizon]
                print(
                    f"{horizon:<8} {row['MAE']:<8.2f} {int(row['N_Forecasts']):<12} {row['Mean_Actual']:<8.1f} {row['Mean_Predicted']:<10.1f}"
                )

        return horizon_metrics

    def compare_with_baselines(self, baseline_results_path="baseline_results.csv"):
        """Compare SARIMA with baseline models"""
        if not hasattr(self, "validation_results"):
            print("Run walk_forward_validation first")
            return

        # Load baseline results
        baseline_df = pd.read_csv(baseline_results_path)

        # Get SARIMA results by horizon
        sarima_metrics = self.validation_results.groupby("horizon")["error"].mean()

        # Create comparison
        comparison = []
        for horizon in range(1, 31):
            baseline_row = baseline_df[baseline_df["Horizon"] == horizon]
            if len(baseline_row) > 0 and horizon in sarima_metrics.index:
                comparison.append(
                    {
                        "Horizon": horizon,
                        "SARIMA_MAE": sarima_metrics[horizon],
                        "Climatology_MAE": baseline_row["Climatology_MAE"].iloc[0],
                        "Seasonal_Naive_MAE": baseline_row["Seasonal_Naive_MAE"].iloc[
                            0
                        ],
                        "Persistence_MAE": baseline_row["Persistence_MAE"].iloc[0],
                    }
                )

        comparison_df = pd.DataFrame(comparison)

        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            comparison_df["Horizon"],
            comparison_df["SARIMA_MAE"],
            "o-",
            label="SARIMA",
            linewidth=2,
            markersize=4,
            color="red",
        )
        ax.plot(
            comparison_df["Horizon"],
            comparison_df["Climatology_MAE"],
            "s-",
            label="Climatology",
            linewidth=2,
            markersize=4,
            color="blue",
        )
        ax.plot(
            comparison_df["Horizon"],
            comparison_df["Seasonal_Naive_MAE"],
            "^-",
            label="Seasonal Naïve",
            linewidth=2,
            markersize=4,
            color="green",
        )
        ax.plot(
            comparison_df["Horizon"],
            comparison_df["Persistence_MAE"],
            "d-",
            label="Persistence",
            linewidth=2,
            markersize=4,
            color="orange",
        )

        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel("Mean Absolute Error (°C)")
        ax.set_title("SARIMA vs Baseline Models")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("sarima_vs_baselines.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Print summary
        print(f"\nSARIMA vs Baselines Summary:")
        print("-" * 40)
        horizons = [1, 7, 14, 30]
        for h in horizons:
            row = comparison_df[comparison_df["Horizon"] == h]
            if len(row) > 0:
                row = row.iloc[0]
                best_baseline = min(
                    row["Climatology_MAE"],
                    row["Seasonal_Naive_MAE"],
                    row["Persistence_MAE"],
                )
                improvement = (
                    (best_baseline - row["SARIMA_MAE"]) / best_baseline
                ) * 100
                print(
                    f"{h:2d} days: SARIMA {row['SARIMA_MAE']:.2f}°C vs Best Baseline {best_baseline:.2f}°C "
                    f"({'↑' if improvement > 0 else '↓'}{improvement:+.1f}%)"
                )

        self.comparison_df = comparison_df
        return comparison_df


def main():
    """Main SARIMA modeling pipeline"""
    # Initialize forecaster
    forecaster = SARIMAForecaster("temagami_features.csv")

    # Check stationarity
    forecaster.check_stationarity(forecaster.temp_series)

    # Apply differencing if needed
    diff_series, diff_orders = forecaster.difference_series(forecaster.temp_series)

    # Plot diagnostics
    forecaster.plot_diagnostics(forecaster.temp_series, "Original Temperature")
    forecaster.plot_diagnostics(diff_series, "Differenced Temperature")

    # Grid search for best parameters (this will take a while)
    print("\nStarting grid search (this may take 10-15 minutes)...")
    best_params, best_seasonal_params = forecaster.grid_search_sarima(
        forecaster.temp_series[:-365], max_order=2
    )  # Hold out last year for validation

    # Walk-forward validation
    validation_results = forecaster.walk_forward_validation(
        test_years=3, forecast_horizon=30
    )

    # Evaluate performance
    horizon_metrics = forecaster.evaluate_sarima_performance()

    # Compare with baselines
    comparison = forecaster.compare_with_baselines()

    return forecaster, validation_results, horizon_metrics


if __name__ == "__main__":
    forecaster, results, metrics = main()
