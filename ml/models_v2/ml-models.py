import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class MLTemperatureForecaster:
    """Machine Learning approach to temperature forecasting"""

    def __init__(self, data_path="temagami_features.csv"):
        """Load the feature-engineered data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded data: {len(self.df)} observations")
        print(
            f"Date range: {self.df.index.min().date()} to {self.df.index.max().date()}"
        )
        print(f"Total features available: {len(self.df.columns)}")

    def prepare_ml_data(self, target_col="t_mean", max_features=15):
        """Prepare data for ML models with careful feature selection"""

        print(f"\nPreparing ML data...")

        # Select safe features (no data leakage)
        safe_features = []

        # Lag features (historical temperatures)
        lag_features = [col for col in self.df.columns if col.startswith("lag_")]
        safe_features.extend(lag_features[:6])  # Limit to first 6 lags

        # Rolling statistics (historical patterns)
        roll_features = [
            col for col in self.df.columns if col.startswith("roll_") and "mean" in col
        ]
        safe_features.extend(roll_features[:4])  # Rolling means only

        # Temporal features (seasonality)
        temporal_features = ["sin_doy", "cos_doy", "sin_month", "cos_month"]
        for feat in temporal_features:
            if feat in self.df.columns:
                safe_features.append(feat)

        # Anomaly features (temperature deviations)
        anomaly_features = [
            col for col in self.df.columns if col.startswith("anomaly_")
        ]
        safe_features.extend(anomaly_features[:2])  # First 2 anomaly features

        # Difference features (rate of change)
        diff_features = [col for col in self.df.columns if col.startswith("diff_")]
        safe_features.extend(diff_features[:2])  # First 2 difference features

        # Limit total features to prevent overfitting
        safe_features = safe_features[:max_features]

        print(f"Selected {len(safe_features)} features:")
        for i, feat in enumerate(safe_features):
            if i < 8:
                print(f"  - {feat}")
            elif i == 8:
                print(f"  ... and {len(safe_features) - 8} more")

        # Create clean dataset
        feature_data = self.df[safe_features + [target_col]].dropna()

        X = feature_data[safe_features].values.astype(np.float32)
        y = feature_data[target_col].values.astype(np.float32)
        dates = feature_data.index

        print(f"Final ML dataset: {len(X)} samples with {X.shape[1]} features")
        print(f"Date range: {dates.min().date()} to {dates.max().date()}")

        self.feature_names = safe_features
        self.ml_data = {"X": X, "y": y, "dates": dates, "feature_names": safe_features}

        return X, y, dates

    def create_ml_models(self):
        """Create diverse ML model configurations"""

        models = {
            "ridge": {
                "name": "Ridge Regression",
                "model": Ridge(alpha=1.0, random_state=42),
                "scale": True,
                "description": "Linear model with L2 regularization",
            },
            "elastic_net": {
                "name": "Elastic Net",
                "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                "scale": True,
                "description": "Linear model with L1+L2 regularization",
            },
            "random_forest": {
                "name": "Random Forest",
                "model": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=4,
                ),
                "scale": False,
                "description": "Ensemble of decision trees",
            },
            "gradient_boosting": {
                "name": "Gradient Boosting",
                "model": GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    subsample=0.8,
                    random_state=42,
                ),
                "scale": False,
                "description": "Sequential ensemble boosting",
            },
            "svr": {
                "name": "Support Vector Regression",
                "model": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
                "scale": True,
                "description": "Non-linear SVM regression",
            },
            "neural_network": {
                "name": "Neural Network",
                "model": MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation="relu",
                    alpha=0.01,
                    learning_rate="adaptive",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
                "scale": True,
                "description": "2-layer neural network",
            },
        }

        print(f"\nCreated {len(models)} ML models:")
        for key, config in models.items():
            print(f"  - {config['name']}: {config['description']}")

        self.ml_models = models
        return models

    def walk_forward_validation(
        self, test_years=2, forecast_horizons=[1, 3, 7, 14, 30]
    ):
        """Comprehensive walk-forward validation for ML models"""

        print(f"\n" + "=" * 60)
        print("ML WALK-FORWARD VALIDATION")
        print("=" * 60)

        # Prepare data
        if not hasattr(self, "ml_data"):
            self.prepare_ml_data()

        if not hasattr(self, "ml_models"):
            self.create_ml_models()

        X, y, dates = self.ml_data["X"], self.ml_data["y"], self.ml_data["dates"]

        # Split data for testing
        test_start_date = dates.max() - pd.Timedelta(days=365 * test_years)
        test_mask = dates > test_start_date

        print(f"Training samples: {(~test_mask).sum()}")
        print(f"Testing samples: {test_mask.sum()}")
        print(f"Split date: {test_start_date.date()}")

        # Test forecast origins (every 3 weeks for manageable computation)
        test_indices = np.where(test_mask)[0][::21]  # Every 21 days

        print(f"Testing from {len(test_indices)} forecast origins...")
        print(f"Forecast horizons: {forecast_horizons} days")

        all_results = []

        # Test each model
        for model_key, model_config in self.ml_models.items():
            print(f"\nTesting {model_config['name']}...")

            model_results = []
            successful_forecasts = 0

            for i, test_idx in enumerate(test_indices):
                if i % 3 == 0:  # Progress updates
                    print(
                        f"  Origin {i+1}/{len(test_indices)}: {dates[test_idx].date()}"
                    )

                try:
                    # Expanding window: train on all data before test point
                    train_mask = np.arange(len(X)) < test_idx

                    if train_mask.sum() < 365:  # Need at least 1 year
                        continue

                    X_train, y_train = X[train_mask], y[train_mask]

                    # Handle scaling
                    if model_config["scale"]:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                    else:
                        X_train_scaled = X_train
                        scaler = None

                    # Fit model
                    model = model_config["model"]

                    # Create fresh instance to avoid fit issues
                    if hasattr(model, "random_state"):
                        model.set_params(random_state=42)

                    model.fit(X_train_scaled, y_train)

                    # Test different forecast horizons
                    for horizon in forecast_horizons:
                        forecast_idx = test_idx + horizon

                        if forecast_idx < len(X):
                            # Get features for forecast point
                            X_forecast = X[forecast_idx : forecast_idx + 1]

                            if model_config["scale"] and scaler is not None:
                                X_forecast_scaled = scaler.transform(X_forecast)
                            else:
                                X_forecast_scaled = X_forecast

                            # Make prediction
                            prediction = model.predict(X_forecast_scaled)[0]
                            actual = y[forecast_idx]

                            # Store result
                            model_results.append(
                                {
                                    "model": model_key,
                                    "origin_date": dates[test_idx],
                                    "forecast_date": dates[forecast_idx],
                                    "horizon": horizon,
                                    "actual": actual,
                                    "predicted": prediction,
                                    "error": abs(actual - prediction),
                                }
                            )

                            successful_forecasts += 1

                except Exception as e:
                    if i % 10 == 0:  # Only print occasional errors to avoid spam
                        print(
                            f"    Error at origin {dates[test_idx].date()}: {str(e)[:50]}..."
                        )
                    continue

            all_results.extend(model_results)
            print(
                f"  Completed: {len(model_results)} forecasts ({successful_forecasts} successful)"
            )

        if not all_results:
            print("⚠ No successful ML forecasts generated!")
            return None

        self.ml_validation_results = pd.DataFrame(all_results)
        print(
            f"\nValidation complete: {len(self.ml_validation_results)} total forecasts"
        )

        return self.ml_validation_results

    def evaluate_ml_performance(self):
        """Comprehensive ML performance evaluation"""

        if (
            not hasattr(self, "ml_validation_results")
            or len(self.ml_validation_results) == 0
        ):
            print("No ML validation results available")
            return None

        print(f"\n" + "=" * 60)
        print("ML MODEL PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Performance by model and horizon
        performance = (
            self.ml_validation_results.groupby(["model", "horizon"])["error"]
            .agg(["mean", "std", "count", "median"])
            .round(3)
        )
        performance.columns = ["MAE", "Std", "Count", "Median"]

        # Display results for each model
        for model_key in self.ml_validation_results["model"].unique():
            model_name = self.ml_models[model_key]["name"]
            print(f"\n{model_name.upper()}:")
            print("-" * 50)
            print(f"{'Horizon':<8} {'MAE (°C)':<10} {'Median':<10} {'Count':<8}")
            print("-" * 50)

            for horizon in sorted(self.ml_validation_results["horizon"].unique()):
                if (model_key, horizon) in performance.index:
                    row = performance.loc[(model_key, horizon)]
                    print(
                        f"{horizon:<8} {row['MAE']:<10.2f} {row['Median']:<10.2f} {int(row['Count']):<8}"
                    )

        self.ml_performance = performance
        return performance

    def compare_ml_models(self):
        """Compare ML models against each other"""

        if not hasattr(self, "ml_validation_results"):
            return None

        print(f"\n" + "=" * 60)
        print("ML MODEL COMPARISON")
        print("=" * 60)

        # Overall performance across all horizons
        model_comparison = (
            self.ml_validation_results.groupby("model")["error"]
            .agg(["mean", "std", "count", "median"])
            .round(3)
        )
        model_comparison.columns = ["Avg_MAE", "Std_MAE", "N_Forecasts", "Median_MAE"]

        # Add model names and sort by performance
        model_comparison["Model_Name"] = [
            self.ml_models[idx]["name"] for idx in model_comparison.index
        ]
        model_comparison = model_comparison.sort_values("Avg_MAE")

        print(f"OVERALL RANKING (Average across all horizons):")
        print("-" * 75)
        print(
            f"{'Rank':<4} {'Model':<25} {'Avg MAE':<10} {'Median':<10} {'N Forecasts':<12}"
        )
        print("-" * 75)

        for rank, (idx, row) in enumerate(model_comparison.iterrows(), 1):
            print(
                f"{rank:<4} {row['Model_Name']:<25} {row['Avg_MAE']:<10.2f} {row['Median_MAE']:<10.2f} {int(row['N_Forecasts']):<12}"
            )

        # Performance by horizon comparison
        print(f"\nPERFORMANCE BY HORIZON:")
        horizons = sorted(self.ml_validation_results["horizon"].unique())

        for horizon in horizons:
            horizon_data = self.ml_validation_results[
                self.ml_validation_results["horizon"] == horizon
            ]
            horizon_comparison = (
                horizon_data.groupby("model")["error"].mean().sort_values()
            )

            print(f"\n{horizon}-day forecasts:")
            for rank, (model_key, mae) in enumerate(horizon_comparison.items(), 1):
                model_name = self.ml_models[model_key]["name"]
                print(f"  {rank}. {model_name}: {mae:.2f}°C")

        self.ml_comparison = model_comparison
        return model_comparison

    def compare_with_baselines(self, baseline_path="baseline_results.csv"):
        """Compare best ML models with baselines and Prophet"""

        try:
            baseline_df = pd.read_csv(baseline_path)
        except FileNotFoundError:
            print("Baseline results not found")
            return

        # Load Prophet results if available
        try:
            prophet_df = pd.read_csv("prophet_performance_summary.csv", index_col=0)
            has_prophet = True
        except FileNotFoundError:
            has_prophet = False
            print("Prophet results not found for comparison")

        if not hasattr(self, "ml_validation_results"):
            print("No ML results to compare")
            return

        # Get best ML model performance by horizon
        best_ml_by_horizon = {}
        for horizon in sorted(self.ml_validation_results["horizon"].unique()):
            horizon_data = self.ml_validation_results[
                self.ml_validation_results["horizon"] == horizon
            ]
            best_performance = horizon_data.groupby("model")["error"].mean().min()
            best_model = horizon_data.groupby("model")["error"].mean().idxmin()

            best_ml_by_horizon[horizon] = {
                "mae": best_performance,
                "model": self.ml_models[best_model]["name"],
            }

        print(f"\n" + "=" * 80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80)

        columns = [
            "Horizon",
            "Best ML",
            "ML Model",
            "Prophet",
            "Climatology",
            "Persistence",
            "Best Overall",
        ]
        if has_prophet:
            print(
                f"{'Horizon':<8} {'Best ML':<10} {'ML Model':<20} {'Prophet':<10} {'Clim':<8} {'Persist':<10} {'Winner':<15}"
            )
        else:
            print(
                f"{'Horizon':<8} {'Best ML':<10} {'ML Model':<20} {'Clim':<8} {'Persist':<10} {'Winner':<15}"
            )
        print("-" * 80)

        overall_improvements = []

        for horizon in [1, 3, 7, 14, 30]:
            if horizon in best_ml_by_horizon:
                ml_mae = best_ml_by_horizon[horizon]["mae"]
                ml_model = best_ml_by_horizon[horizon]["model"]

                baseline_row = baseline_df[baseline_df["Horizon"] == horizon]

                if len(baseline_row) > 0:
                    clim_mae = baseline_row["Climatology_MAE"].iloc[0]
                    persist_mae = baseline_row["Persistence_MAE"].iloc[0]

                    # Get Prophet performance if available
                    if has_prophet and horizon in prophet_df.index:
                        prophet_mae = prophet_df.loc[horizon, "mae"]
                        all_models = {
                            "ML": ml_mae,
                            "Prophet": prophet_mae,
                            "Climatology": clim_mae,
                            "Persistence": persist_mae,
                        }
                    else:
                        prophet_mae = None
                        all_models = {
                            "ML": ml_mae,
                            "Climatology": clim_mae,
                            "Persistence": persist_mae,
                        }

                    # Find best overall
                    best_overall = min(all_models.items(), key=lambda x: x[1])
                    winner = best_overall[0]

                    # Calculate improvement over baselines
                    best_baseline = min(clim_mae, persist_mae)
                    ml_improvement = ((best_baseline - ml_mae) / best_baseline) * 100
                    overall_improvements.append(ml_improvement)

                    # Print row
                    if has_prophet and prophet_mae is not None:
                        print(
                            f"{horizon:<8} {ml_mae:<10.2f} {ml_model:<20} {prophet_mae:<10.2f} {clim_mae:<8.2f} {persist_mae:<10.2f} {winner:<15}"
                        )
                    else:
                        print(
                            f"{horizon:<8} {ml_mae:<10.2f} {ml_model:<20} {clim_mae:<8.2f} {persist_mae:<10.2f} {winner:<15}"
                        )

        if overall_improvements:
            avg_improvement = np.mean(overall_improvements)
            print(f"\nML Models vs Baselines:")
            print(f"Average improvement over best baseline: {avg_improvement:+.1f}%")

    def analyze_feature_importance(self):
        """Analyze feature importance from tree-based models"""

        if not hasattr(self, "ml_data"):
            print("ML data not prepared")
            return

        print(f"\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        X, y, dates = self.ml_data["X"], self.ml_data["y"], self.ml_data["dates"]
        feature_names = self.ml_data["feature_names"]

        # Fit Random Forest on full dataset for feature importance
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=4
        )

        rf_model.fit(X, y)

        # Get feature importance
        importance = rf_model.feature_importances_
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print(f"TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        print(f"{'Feature':<20} {'Importance':<12} {'Type':<15}")
        print("-" * 40)

        for i, (feature, imp) in enumerate(feature_importance[:10]):
            # Categorize feature type
            if feature.startswith("lag_"):
                feat_type = "Historical"
            elif feature.startswith("roll_"):
                feat_type = "Rolling Stat"
            elif feature.startswith("sin_") or feature.startswith("cos_"):
                feat_type = "Seasonal"
            elif feature.startswith("anomaly_"):
                feat_type = "Anomaly"
            elif feature.startswith("diff_"):
                feat_type = "Trend"
            else:
                feat_type = "Other"

            print(f"{feature:<20} {imp:<12.3f} {feat_type:<15}")

        return feature_importance

    def plot_ml_results(self):
        """Comprehensive ML results visualization"""

        if not hasattr(self, "ml_validation_results"):
            print("No ML results to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Performance by horizon for each model
        models = self.ml_validation_results["model"].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

        for model, color in zip(models, colors):
            model_data = self.ml_validation_results[
                self.ml_validation_results["model"] == model
            ]
            horizon_performance = model_data.groupby("horizon")["error"].mean()

            axes[0, 0].plot(
                horizon_performance.index,
                horizon_performance.values,
                "o-",
                label=self.ml_models[model]["name"],
                color=color,
                linewidth=2,
                markersize=6,
            )

        axes[0, 0].set_xlabel("Forecast Horizon (days)")
        axes[0, 0].set_ylabel("Mean Absolute Error (°C)")
        axes[0, 0].set_title("ML Model Performance by Horizon")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Model comparison (average performance)
        if hasattr(self, "ml_comparison"):
            model_names = [
                self.ml_models[idx]["name"] for idx in self.ml_comparison.index
            ]
            avg_maes = self.ml_comparison["Avg_MAE"].values

            bars = axes[0, 1].bar(
                range(len(model_names)),
                avg_maes,
                color=colors[: len(model_names)],
                alpha=0.7,
            )
            axes[0, 1].set_xlabel("Models")
            axes[0, 1].set_ylabel("Average MAE (°C)")
            axes[0, 1].set_title("Overall ML Model Comparison")
            axes[0, 1].set_xticks(range(len(model_names)))
            axes[0, 1].set_xticklabels(
                [name.replace(" ", "\n") for name in model_names], rotation=0
            )
            axes[0, 1].grid(True, alpha=0.3)

            # Add values on bars
            for bar, mae in zip(bars, avg_maes):
                axes[0, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{mae:.2f}",
                    ha="center",
                    va="bottom",
                )

        # Plot 3: Error distribution
        axes[0, 2].hist(
            self.ml_validation_results["error"],
            bins=50,
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )
        axes[0, 2].set_xlabel("Absolute Error (°C)")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_title("Distribution of ML Forecast Errors")
        axes[0, 2].grid(True, alpha=0.3)

        mean_error = self.ml_validation_results["error"].mean()
        median_error = self.ml_validation_results["error"].median()
        axes[0, 2].axvline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.2f}°C"
        )
        axes[0, 2].axvline(
            median_error,
            color="orange",
            linestyle="--",
            label=f"Median: {median_error:.2f}°C",
        )
        axes[0, 2].legend()

        # Plot 4: Feature importance (if available)
        if hasattr(self, "ml_data"):
            feature_importance = self.analyze_feature_importance()

            if feature_importance:
                top_features = feature_importance[:8]  # Top 8 features
                features, importances = zip(*top_features)

                axes[1, 0].barh(
                    range(len(features)), importances, alpha=0.7, color="green"
                )
                axes[1, 0].set_xlabel("Feature Importance")
                axes[1, 0].set_title("Top Features (Random Forest)")
                axes[1, 0].set_yticks(range(len(features)))
                axes[1, 0].set_yticklabels(features)
                axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Actual vs Predicted scatter
        sample_results = self.ml_validation_results.sample(
            min(500, len(self.ml_validation_results))
        )

        axes[1, 1].scatter(
            sample_results["actual"],
            sample_results["predicted"],
            alpha=0.5,
            s=20,
            c=sample_results["horizon"],
            cmap="viridis",
        )

        # Perfect prediction line
        min_temp = min(
            sample_results["actual"].min(), sample_results["predicted"].min()
        )
        max_temp = max(
            sample_results["actual"].max(), sample_results["predicted"].max()
        )
        axes[1, 1].plot([min_temp, max_temp], [min_temp, max_temp], "r--", alpha=0.8)

        axes[1, 1].set_xlabel("Actual Temperature (°C)")
        axes[1, 1].set_ylabel("Predicted Temperature (°C)")
        axes[1, 1].set_title("Actual vs Predicted (Sample)")
        axes[1, 1].grid(True, alpha=0.3)

        # Add colorbar for horizon
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label("Forecast Horizon (days)")

        # Plot 6: Performance by season
        sample_results["month"] = pd.to_datetime(
            sample_results["forecast_date"]
        ).dt.month
        seasonal_performance = sample_results.groupby("month")["error"].mean()

        axes[1, 2].plot(
            seasonal_performance.index,
            seasonal_performance.values,
            "o-",
            linewidth=2,
            markersize=6,
            color="red",
        )
        axes[1, 2].set_xlabel("Month")
        axes[1, 2].set_ylabel("Mean Absolute Error (°C)")
        axes[1, 2].set_title("ML Performance by Season")
        axes[1, 2].set_xticks(range(1, 13))
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("ml_comprehensive_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Saved comprehensive ML analysis: ml_comprehensive_analysis.png")


def main():
    """Main ML forecasting pipeline"""
    print("=" * 60)
    print("MACHINE LEARNING TEMPERATURE FORECASTING")
    print("=" * 60)

    # Initialize forecaster
    forecaster = MLTemperatureForecaster("temagami_features.csv")

    # Prepare data with careful feature selection
    X, y, dates = forecaster.prepare_ml_data(max_features=15)

    # Create diverse ML models
    models = forecaster.create_ml_models()

    # Run comprehensive validation
    print(f"\nStarting walk-forward validation...")
    validation_results = forecaster.walk_forward_validation(
        test_years=2, forecast_horizons=[1, 3, 7, 14, 30]
    )

    if validation_results is not None:
        # Evaluate performance
        performance = forecaster.evaluate_ml_performance()

        # Compare models
        model_comparison = forecaster.compare_ml_models()

        # Compare with baselines and Prophet
        forecaster.compare_with_baselines()

        # Analyze feature importance
        feature_importance = forecaster.analyze_feature_importance()

        # Create comprehensive plots
        forecaster.plot_ml_results()

        # Save results
        validation_results.to_csv("ml_validation_results.csv", index=False)

        if hasattr(forecaster, "ml_comparison"):
            forecaster.ml_comparison.to_csv("ml_model_comparison.csv")

        print(f"\n" + "=" * 60)
        print("ML ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"✓ Tested {len(models)} different ML algorithms")
        print(f"✓ Generated {len(validation_results)} forecasts")
        print(f"✓ Comprehensive comparison with baselines and Prophet")
        print(f"✓ Results saved to ml_validation_results.csv")

        return forecaster, validation_results, performance

    else:
        print("ML validation failed")
        return None, None, None


if __name__ == "__main__":
    forecaster, results, performance = main()
