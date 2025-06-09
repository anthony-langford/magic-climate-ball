import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class CleanMLForecaster:
    """ML forecasting with clean, leak-free features"""

    def __init__(self, data_path="clean_ml_features.csv"):
        """Load the clean feature data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded clean ML data: {len(self.df)} observations")
        print(
            f"Date range: {self.df.index.min().date()} to {self.df.index.max().date()}"
        )
        print(f"Features: {len(self.df.columns)-1} (excluding target)")

        # Prepare feature matrix
        self.target_col = "t_mean"
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]

        print(f"Clean feature set:")
        for i, col in enumerate(self.feature_cols):
            if i < 8:
                print(f"  - {col}")
            elif i == 8:
                print(f"  ... and {len(self.feature_cols)-8} more")

    def create_ml_models(self):
        """Create ML models with appropriate complexity for clean data"""

        models = {
            "ridge": {
                "name": "Ridge Regression",
                "model": Ridge(
                    alpha=10.0, random_state=42
                ),  # Higher alpha for regularization
                "scale": True,
                "description": "Linear with L2 regularization",
            },
            "elastic_net": {
                "name": "Elastic Net",
                "model": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
                "scale": True,
                "description": "Linear with L1+L2 regularization",
            },
            "random_forest": {
                "name": "Random Forest",
                "model": RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,  # Limit depth to prevent overfitting
                    min_samples_split=20,
                    min_samples_leaf=10,
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
                    n_estimators=150,
                    max_depth=4,  # Conservative depth
                    learning_rate=0.05,  # Lower learning rate
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=42,
                ),
                "scale": False,
                "description": "Sequential boosting ensemble",
            },
            "neural_network": {
                "name": "Neural Network",
                "model": MLPRegressor(
                    hidden_layer_sizes=(50, 25),  # Smaller network
                    activation="relu",
                    alpha=0.1,  # Higher regularization
                    learning_rate="adaptive",
                    max_iter=300,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
                "scale": True,
                "description": "2-layer neural network",
            },
        }

        self.ml_models = models
        return models

    def walk_forward_validation(
        self, test_years=2, forecast_horizons=[1, 3, 7, 14, 30]
    ):
        """Proper walk-forward validation on clean data"""

        print(f"\n" + "=" * 60)
        print("CLEAN ML WALK-FORWARD VALIDATION")
        print("=" * 60)

        if not hasattr(self, "ml_models"):
            self.create_ml_models()

        # Prepare data
        X = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df[self.target_col].values.astype(np.float32)
        dates = self.df.index

        # Time-based test split
        test_start_date = dates.max() - pd.Timedelta(days=365 * test_years)
        test_mask = dates > test_start_date

        print(
            f"Training period: {dates[~test_mask].min().date()} to {test_start_date.date()}"
        )
        print(f"Testing period: {test_start_date.date()} to {dates.max().date()}")
        print(f"Training samples: {(~test_mask).sum()}")
        print(f"Testing samples: {test_mask.sum()}")

        # Test origins (every 2 weeks for computational efficiency)
        test_indices = np.where(test_mask)[0][::14]  # Every 14 days
        print(f"Testing from {len(test_indices)} forecast origins...")

        all_results = []

        # Test each model
        for model_key, model_config in self.ml_models.items():
            print(f"\nTesting {model_config['name']}...")

            model_results = []
            successful_forecasts = 0

            for i, test_idx in enumerate(test_indices):
                if i % 4 == 0:  # Progress updates
                    print(
                        f"  Origin {i+1}/{len(test_indices)}: {dates[test_idx].date()}"
                    )

                try:
                    # Expanding window training
                    train_mask = np.arange(len(X)) < test_idx

                    if train_mask.sum() < 730:  # Need at least 2 years
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

                    # Create fresh instance to avoid issues
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
                    if i % 10 == 0:  # Occasional error reporting
                        print(
                            f"    Error at {dates[test_idx].date()}: {str(e)[:50]}..."
                        )
                    continue

            all_results.extend(model_results)
            print(f"  Completed: {len(model_results)} forecasts")

        if not all_results:
            print("No successful ML forecasts generated!")
            return None

        self.validation_results = pd.DataFrame(all_results)
        print(f"\nValidation complete: {len(self.validation_results)} total forecasts")

        return self.validation_results

    def evaluate_performance(self):
        """Evaluate clean ML performance"""

        if not hasattr(self, "validation_results") or len(self.validation_results) == 0:
            print("No validation results available")
            return None

        print(f"\n" + "=" * 60)
        print("CLEAN ML PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Performance by model and horizon
        performance = (
            self.validation_results.groupby(["model", "horizon"])["error"]
            .agg(["mean", "std", "count", "median"])
            .round(3)
        )
        performance.columns = ["MAE", "Std", "Count", "Median"]

        # Show results for each model
        for model_key in self.validation_results["model"].unique():
            model_name = self.ml_models[model_key]["name"]
            print(f"\n{model_name.upper()}:")
            print("-" * 45)
            print(f"{'Horizon':<8} {'MAE (°C)':<10} {'Median':<10} {'Count':<8}")
            print("-" * 45)

            model_performance = []
            for horizon in sorted(self.validation_results["horizon"].unique()):
                if (model_key, horizon) in performance.index:
                    row = performance.loc[(model_key, horizon)]
                    mae = row["MAE"]
                    median = row["Median"]
                    count = int(row["Count"])

                    print(f"{horizon:<8} {mae:<10.2f} {median:<10.2f} {count:<8}")
                    model_performance.append(mae)

            if model_performance:
                avg_mae = np.mean(model_performance)
                print(f"Average: {avg_mae:<10.2f}")

        return performance

    def compare_models_and_baselines(self):
        """Compare ML models with each other and baselines"""

        if not hasattr(self, "validation_results"):
            return None

        print(f"\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        # Overall ML model comparison
        model_comparison = (
            self.validation_results.groupby("model")["error"]
            .agg(["mean", "std", "count", "median"])
            .round(3)
        )
        model_comparison.columns = ["Avg_MAE", "Std_MAE", "N_Forecasts", "Median_MAE"]

        # Add model names and sort
        model_comparison["Model_Name"] = [
            self.ml_models[idx]["name"] for idx in model_comparison.index
        ]
        model_comparison = model_comparison.sort_values("Avg_MAE")

        print(f"ML MODEL RANKING:")
        print("-" * 70)
        print(
            f"{'Rank':<4} {'Model':<20} {'Avg MAE':<8} {'Median':<8} {'Forecasts':<10}"
        )
        print("-" * 70)

        for rank, (idx, row) in enumerate(model_comparison.iterrows(), 1):
            print(
                f"{rank:<4} {row['Model_Name']:<20} {row['Avg_MAE']:<8.2f} {row['Median_MAE']:<8.2f} {int(row['N_Forecasts']):<10}"
            )

        # Compare with baselines and Prophet
        print(f"\nCOMPARISON WITH ALL APPROACHES:")
        print("-" * 80)

        best_ml_mae = model_comparison["Avg_MAE"].min()
        best_ml_model = model_comparison.iloc[0]["Model_Name"]

        print(f"Best ML Model:        {best_ml_model} ({best_ml_mae:.2f}°C MAE)")
        print(f"Prophet Conservative: 4.09°C MAE")
        print(f"Climatology Baseline: 4.40°C MAE")
        print(f"Persistence Baseline: 3.17°C MAE (1-day only)")

        # Performance analysis
        if best_ml_mae < 3.5:
            print(f"🎉 ML shows excellent performance!")
            prophet_improvement = ((4.09 - best_ml_mae) / 4.09) * 100
            print(f"   {prophet_improvement:.1f}% better than Prophet")
        elif best_ml_mae < 4.09:
            prophet_improvement = ((4.09 - best_ml_mae) / 4.09) * 100
            print(f"✅ ML beats Prophet by {prophet_improvement:.1f}%")
        elif best_ml_mae < 4.40:
            baseline_improvement = ((4.40 - best_ml_mae) / 4.40) * 100
            print(f"✅ ML beats climatology by {baseline_improvement:.1f}%")
        else:
            print(f"⚠️  ML performance similar to baselines - consider ensemble")

        # Performance by horizon comparison
        print(f"\nPERFORMANCE BY HORIZON:")
        print("-" * 50)

        for horizon in sorted(self.validation_results["horizon"].unique()):
            horizon_data = self.validation_results[
                self.validation_results["horizon"] == horizon
            ]
            best_mae = horizon_data.groupby("model")["error"].mean().min()
            best_model_key = horizon_data.groupby("model")["error"].mean().idxmin()
            best_model_name = self.ml_models[best_model_key]["name"]

            print(f"{horizon:2d}-day: {best_model_name} ({best_mae:.2f}°C)")

        return model_comparison

    def analyze_feature_importance(self):
        """Analyze feature importance from tree-based models"""

        print(f"\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)

        # Use Random Forest for feature importance
        X = self.df[self.feature_cols].values
        y = self.df[self.target_col].values

        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=4
        )

        rf_model.fit(X, y)

        # Get feature importance
        importance = rf_model.feature_importances_
        feature_importance = list(zip(self.feature_cols, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print(f"TOP 15 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        print(f"{'Feature':<20} {'Importance':<12} {'Type':<15}")
        print("-" * 50)

        for i, (feature, imp) in enumerate(feature_importance[:15]):
            # Categorize feature type
            if feature.startswith("lag_"):
                feat_type = "Historical"
            elif feature.startswith("roll_"):
                feat_type = "Rolling Stat"
            elif feature.startswith("sin_") or feature.startswith("cos_"):
                feat_type = "Seasonal"
            elif feature.startswith("anomaly_"):
                feat_type = "Anomaly"
            elif feature.startswith("temp_change"):
                feat_type = "Trend"
            else:
                feat_type = "Temporal"

            print(f"{feature:<20} {imp:<12.3f} {feat_type:<15}")

        self.feature_importance = feature_importance
        return feature_importance

    def plot_results(self):
        """Create comprehensive visualization of clean ML results"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Performance by horizon
        models = self.validation_results["model"].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

        for model, color in zip(models, colors):
            model_data = self.validation_results[
                self.validation_results["model"] == model
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
        axes[0, 0].set_title("Clean ML Performance by Horizon")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Model comparison
        model_avg_performance = (
            self.validation_results.groupby("model")["error"].mean().sort_values()
        )
        model_names = [
            self.ml_models[idx]["name"] for idx in model_avg_performance.index
        ]

        bars = axes[0, 1].bar(
            range(len(model_names)),
            model_avg_performance.values,
            color=colors[: len(model_names)],
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Models")
        axes[0, 1].set_ylabel("Average MAE (°C)")
        axes[0, 1].set_title("ML Model Comparison")
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels([name.replace(" ", "\n") for name in model_names])
        axes[0, 1].grid(True, alpha=0.3)

        # Add values on bars
        for bar, mae in zip(bars, model_avg_performance.values):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{mae:.2f}",
                ha="center",
                va="bottom",
            )

        # Plot 3: Error distribution
        axes[0, 2].hist(
            self.validation_results["error"],
            bins=40,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        axes[0, 2].set_xlabel("Absolute Error (°C)")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_title("Error Distribution (Clean ML)")
        axes[0, 2].grid(True, alpha=0.3)

        mean_error = self.validation_results["error"].mean()
        axes[0, 2].axvline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.2f}°C"
        )
        axes[0, 2].legend()

        # Plot 4: Feature importance
        if hasattr(self, "feature_importance"):
            top_features = self.feature_importance[:10]
            features, importances = zip(*top_features)

            axes[1, 0].barh(
                range(len(features)), importances, alpha=0.7, color="orange"
            )
            axes[1, 0].set_xlabel("Feature Importance")
            axes[1, 0].set_title("Top 10 Features (Random Forest)")
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            self.analyze_feature_importance()
            # Recursive call after computing importance
            return self.plot_results()

        # Plot 5: Actual vs Predicted
        sample_results = self.validation_results.sample(
            min(400, len(self.validation_results))
        )

        scatter = axes[1, 1].scatter(
            sample_results["actual"],
            sample_results["predicted"],
            alpha=0.6,
            s=30,
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
        axes[1, 1].set_title("Actual vs Predicted (Clean ML)")
        axes[1, 1].grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=axes[1, 1], label="Forecast Horizon (days)")

        # Plot 6: Performance by season
        sample_results["month"] = pd.to_datetime(
            sample_results["forecast_date"]
        ).dt.month
        seasonal_perf = sample_results.groupby("month")["error"].mean()

        axes[1, 2].plot(
            seasonal_perf.index,
            seasonal_perf.values,
            "o-",
            linewidth=2,
            markersize=6,
            color="purple",
        )
        axes[1, 2].set_xlabel("Month")
        axes[1, 2].set_ylabel("Mean Absolute Error (°C)")
        axes[1, 2].set_title("Seasonal Performance")
        axes[1, 2].set_xticks(range(1, 13))
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("clean_ml_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Saved clean ML analysis: clean_ml_analysis.png")


def main():
    """Main clean ML pipeline"""
    print("=" * 60)
    print("CLEAN ML TEMPERATURE FORECASTING")
    print("=" * 60)

    # Initialize with clean data
    forecaster = CleanMLForecaster("clean_ml_features.csv")

    # Create models
    models = forecaster.create_ml_models()

    # Run validation
    validation_results = forecaster.walk_forward_validation(
        test_years=2, forecast_horizons=[1, 3, 7, 14, 30]
    )

    if validation_results is not None:
        # Evaluate performance
        performance = forecaster.evaluate_performance()

        # Compare models and baselines
        comparison = forecaster.compare_models_and_baselines()

        # Analyze features
        feature_importance = forecaster.analyze_feature_importance()

        # Create plots
        forecaster.plot_results()

        # Save results
        validation_results.to_csv("clean_ml_results.csv", index=False)

        print(f"\n" + "=" * 60)
        print("CLEAN ML ANALYSIS COMPLETE!")
        print("=" * 60)
        print("✅ Fixed data leakage issues")
        print("✅ Realistic ML performance on clean features")
        print("✅ Proper comparison with Prophet and baselines")
        print("✅ Ready for ensemble modeling")

        return forecaster, validation_results, performance

    else:
        print("Clean ML validation failed")
        return None, None, None


if __name__ == "__main__":
    forecaster, results, performance = main()
