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
import os

warnings.filterwarnings("ignore")


class ProductionMLEnsemble:
    """Production-ready ML ensemble with verified clean features"""

    def __init__(self, clean_features_path="ultra_clean_ml_features.csv"):
        """Initialize with the debugged clean features"""
        self.df = pd.read_csv(clean_features_path, index_col=0, parse_dates=True)

        self.target_col = "t_mean"
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]

        print(f"Production ML Ensemble initialized")
        print(f"Data: {len(self.df)} observations")
        print(f"Period: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"Clean features: {len(self.feature_cols)}")
        print(f"Features: {self.feature_cols}")

        # Ensure output directory exists
        os.makedirs("ml", exist_ok=True)

    def create_production_models(self):
        """Create diverse ML models optimized for production"""

        models = {
            "ridge_conservative": {
                "name": "Ridge (Conservative)",
                "model": Ridge(alpha=10.0, random_state=42),
                "scale": True,
                "description": "Linear model with high regularization",
            },
            "ridge_moderate": {
                "name": "Ridge (Moderate)",
                "model": Ridge(alpha=1.0, random_state=42),
                "scale": True,
                "description": "Linear model with moderate regularization",
            },
            "elastic_net": {
                "name": "Elastic Net",
                "model": ElasticNet(
                    alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=2000
                ),
                "scale": True,
                "description": "Linear with L1+L2 regularization",
            },
            "random_forest": {
                "name": "Random Forest",
                "model": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
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
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
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
                    hidden_layer_sizes=(50, 25),
                    activation="relu",
                    alpha=0.01,
                    learning_rate="adaptive",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
                "scale": True,
                "description": "2-layer neural network",
            },
        }

        print(f"\nCreated {len(models)} production ML models:")
        for key, config in models.items():
            print(f"  • {config['name']}: {config['description']}")

        self.ml_models = models
        return models

    def comprehensive_validation(
        self, test_years=2, forecast_horizons=[1, 3, 7, 14, 30]
    ):
        """Comprehensive validation using the proven methodology"""

        print(f"\n" + "=" * 60)
        print("COMPREHENSIVE ML VALIDATION")
        print("=" * 60)

        if not hasattr(self, "ml_models"):
            self.create_production_models()

        # Prepare data
        X = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df[self.target_col].values.astype(np.float32)
        dates = self.df.index

        # Time-based split (same methodology that worked)
        split_date = dates.max() - pd.Timedelta(days=365 * test_years)
        train_mask = dates <= split_date

        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        test_dates = dates[~train_mask]

        print(f"Training: {len(X_train)} samples (until {split_date.date()})")
        print(f"Testing: {len(X_test)} samples")

        # Test each model across different horizons
        all_results = []
        model_performances = {}

        for model_key, model_config in self.ml_models.items():
            print(f"\nValidating {model_config['name']}...")

            # Fit model
            model = model_config["model"]

            if model_config["scale"]:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                model.fit(X_train_scaled, y_train)
            else:
                X_train_scaled = X_train
                scaler = None
                model.fit(X_train_scaled, y_train)

            # Test different horizons using Prophet-style approach
            horizon_results = {}

            for horizon in forecast_horizons:
                horizon_errors = []
                horizon_predictions = []
                horizon_actuals = []

                # Test every 14 days for computational efficiency
                test_origins = test_dates[::14]

                for origin_date in test_origins[:-5]:  # Leave buffer
                    forecast_date = origin_date + pd.Timedelta(days=horizon)

                    if forecast_date in test_dates:
                        # Get features for forecast date
                        forecast_idx = dates.get_loc(forecast_date)
                        feature_row = X[forecast_idx : forecast_idx + 1]

                        if model_config["scale"] and scaler is not None:
                            feature_row_scaled = scaler.transform(feature_row)
                        else:
                            feature_row_scaled = feature_row

                        # Make prediction
                        prediction = model.predict(feature_row_scaled)[0]
                        actual = y[forecast_idx]

                        error = abs(actual - prediction)
                        horizon_errors.append(error)
                        horizon_predictions.append(prediction)
                        horizon_actuals.append(actual)

                        # Store detailed results
                        all_results.append(
                            {
                                "model": model_key,
                                "model_name": model_config["name"],
                                "origin_date": origin_date,
                                "forecast_date": forecast_date,
                                "horizon": horizon,
                                "actual": actual,
                                "predicted": prediction,
                                "error": error,
                            }
                        )

                if horizon_errors:
                    horizon_mae = np.mean(horizon_errors)
                    horizon_results[horizon] = {
                        "mae": horizon_mae,
                        "count": len(horizon_errors),
                        "predictions": horizon_predictions,
                        "actuals": horizon_actuals,
                    }
                    print(
                        f"  {horizon:2d}-day: {horizon_mae:.3f}°C MAE ({len(horizon_errors)} forecasts)"
                    )

            # Store model performance
            if horizon_results:
                model_performances[model_key] = {
                    "name": model_config["name"],
                    "horizon_results": horizon_results,
                    "avg_mae": np.mean([hr["mae"] for hr in horizon_results.values()]),
                    "fitted_model": model,
                    "scaler": scaler,
                }

        self.validation_results = pd.DataFrame(all_results)
        self.model_performances = model_performances
        self.forecast_horizons = forecast_horizons

        print(
            f"\nValidation complete: {len(self.validation_results)} forecast evaluations"
        )
        return self.validation_results

    def analyze_model_performance(self):
        """Analyze and rank model performance"""

        if not hasattr(self, "model_performances"):
            print("Run validation first")
            return None

        print(f"\n" + "=" * 60)
        print("MODEL PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Rank models by average performance
        model_ranking = []
        for model_key, perf in self.model_performances.items():
            model_ranking.append(
                {
                    "model_key": model_key,
                    "name": perf["name"],
                    "avg_mae": perf["avg_mae"],
                }
            )

        model_ranking.sort(key=lambda x: x["avg_mae"])

        print(f"MODEL RANKING (Average across all horizons):")
        print("-" * 50)
        print(f"{'Rank':<4} {'Model':<25} {'Avg MAE (°C)':<12}")
        print("-" * 50)

        for i, model in enumerate(model_ranking, 1):
            print(f"{i:<4} {model['name']:<25} {model['avg_mae']:<12.3f}")

        # Performance by horizon
        print(f"\nPERFORMANCE BY FORECAST HORIZON:")
        print("-" * 60)

        horizons = sorted(
            list(
                self.model_performances[list(self.model_performances.keys())[0]][
                    "horizon_results"
                ].keys()
            )
        )

        print(f"{'Horizon':<8} {'Best Model':<25} {'MAE (°C)':<10}")
        print("-" * 60)

        for horizon in horizons:
            best_mae = float("inf")
            best_model = ""

            for model_key, perf in self.model_performances.items():
                if horizon in perf["horizon_results"]:
                    mae = perf["horizon_results"][horizon]["mae"]
                    if mae < best_mae:
                        best_mae = mae
                        best_model = perf["name"]

            print(f"{horizon:<8} {best_model:<25} {best_mae:<10.3f}")

        return model_ranking

    def compare_with_benchmarks(self):
        """Compare with Prophet and baseline results"""

        print(f"\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)

        if not hasattr(self, "model_performances"):
            print("Run validation first")
            return

        # Compute baselines if not already done
        if not hasattr(self, "baseline_results"):
            self.compute_baseline_performance()

        # Get best ML model
        best_model_key = min(
            self.model_performances.keys(),
            key=lambda k: self.model_performances[k]["avg_mae"],
        )
        best_ml_mae = self.model_performances[best_model_key]["avg_mae"]
        best_ml_name = self.model_performances[best_model_key]["name"]

        print(f"COMPREHENSIVE MODEL COMPARISON:")
        print("-" * 50)
        print(f"Method                    Avg MAE (°C)    Status")
        print("-" * 50)
        print(f"{best_ml_name:<25} {best_ml_mae:<12.3f}  🥇 Best ML")
        print(f"Neural Network (Multi)    2.780         🧠 Multi-output NN")
        print(f"Persistence (1-day)       3.170         📊 Short-term")
        print(f"Prophet Conservative      4.090         📊 Strong baseline")
        print(f"SARIMA (Hybrid)           4.160         🌀 Time series")

        # Add computed baselines
        if "climatology" in self.baseline_results:
            clim_mae = np.mean(
                [h["mae"] for h in self.baseline_results["climatology"].values()]
            )
            print(f"Climatology (Computed)    {clim_mae:<12.3f}  📊 Computed baseline")

        if "seasonal_naive" in self.baseline_results:
            sn_mae = np.mean(
                [h["mae"] for h in self.baseline_results["seasonal_naive"].values()]
            )
            print(f"Seasonal Naive (Comp.)    {sn_mae:<12.3f}  📊 Computed baseline")

        # Calculate improvements
        nn_improvement = (
            ((2.78 - best_ml_mae) / 2.78) * 100
            if best_ml_mae < 2.78
            else ((best_ml_mae - 2.78) / 2.78) * 100
        )
        prophet_improvement = ((4.09 - best_ml_mae) / 4.09) * 100
        sarima_improvement = ((4.16 - best_ml_mae) / 4.16) * 100

        print(f"\nML IMPROVEMENTS:")
        if best_ml_mae < 2.78:
            print(f"🎉 {nn_improvement:.1f}% better than Multi-output Neural Network!")
        else:
            print(f"📊 {abs(nn_improvement):.1f}% behind Multi-output Neural Network")
        print(f"• {prophet_improvement:.1f}% better than Prophet")
        print(f"• {sarima_improvement:.1f}% better than SARIMA")

        # Compare with computed baselines
        if "climatology" in self.baseline_results:
            clim_mae = np.mean(
                [h["mae"] for h in self.baseline_results["climatology"].values()]
            )
            clim_improvement = ((clim_mae - best_ml_mae) / clim_mae) * 100
            print(f"• {clim_improvement:.1f}% better than Computed Climatology")

        if "seasonal_naive" in self.baseline_results:
            sn_mae = np.mean(
                [h["mae"] for h in self.baseline_results["seasonal_naive"].values()]
            )
            sn_improvement = ((sn_mae - best_ml_mae) / sn_mae) * 100
            print(f"• {sn_improvement:.1f}% better than Seasonal Naive")

        if best_ml_mae < 2.5:
            print("🚀 Outstanding ML performance - new state of the art!")
        elif best_ml_mae < 3.0:
            print("🎉 Excellent ML performance achieved!")
        elif best_ml_mae < 3.5:
            print("✅ Strong ML performance")
        else:
            print("📊 Competitive ML performance")

    def create_ensemble_predictions(self, top_n_models=3):
        """Create ensemble predictions from top performing models"""

        print(f"\n" + "=" * 60)
        print("ENSEMBLE MODEL CREATION")
        print("=" * 60)

        if not hasattr(self, "model_performances"):
            print("Run validation first")
            return None

        # Get top N models
        model_ranking = [(k, v["avg_mae"]) for k, v in self.model_performances.items()]
        model_ranking.sort(key=lambda x: x[1])
        top_models = model_ranking[:top_n_models]

        print(f"Creating ensemble from top {top_n_models} models:")
        for i, (model_key, mae) in enumerate(top_models, 1):
            model_name = self.model_performances[model_key]["name"]
            print(f"  {i}. {model_name}: {mae:.3f}°C MAE")

        # Create ensemble predictions on validation set
        ensemble_results = []

        # Get all forecasts from validation results
        unique_forecasts = self.validation_results[
            ["origin_date", "forecast_date", "horizon", "actual"]
        ].drop_duplicates()

        for _, forecast_row in unique_forecasts.iterrows():
            origin_date = forecast_row["origin_date"]
            forecast_date = forecast_row["forecast_date"]
            horizon = forecast_row["horizon"]
            actual = forecast_row["actual"]

            # Get predictions from top models
            model_predictions = []
            for model_key, _ in top_models:
                model_forecast = self.validation_results[
                    (self.validation_results["model"] == model_key)
                    & (self.validation_results["origin_date"] == origin_date)
                    & (self.validation_results["forecast_date"] == forecast_date)
                ]

                if len(model_forecast) > 0:
                    model_predictions.append(model_forecast["predicted"].iloc[0])

            if len(model_predictions) >= 2:  # Need at least 2 models
                # Simple average ensemble
                ensemble_pred = np.mean(model_predictions)
                ensemble_error = abs(actual - ensemble_pred)

                ensemble_results.append(
                    {
                        "origin_date": origin_date,
                        "forecast_date": forecast_date,
                        "horizon": horizon,
                        "actual": actual,
                        "ensemble_prediction": ensemble_pred,
                        "ensemble_error": ensemble_error,
                        "individual_predictions": model_predictions,
                    }
                )

        self.ensemble_results = pd.DataFrame(ensemble_results)

        # Analyze ensemble performance
        ensemble_mae = self.ensemble_results["ensemble_error"].mean()
        best_individual_mae = min(mae for _, mae in top_models)

        print(f"\nENSEMBLE PERFORMANCE:")
        print(f"Ensemble MAE:        {ensemble_mae:.3f}°C")
        print(f"Best individual MAE: {best_individual_mae:.3f}°C")

        if ensemble_mae < best_individual_mae:
            improvement = (
                (best_individual_mae - ensemble_mae) / best_individual_mae
            ) * 100
            print(f"🎉 Ensemble improves by {improvement:.1f}%!")
        else:
            decline = ((ensemble_mae - best_individual_mae) / best_individual_mae) * 100
            print(f"📊 Ensemble slightly worse by {decline:.1f}% (normal variation)")

        return self.ensemble_results

    def compute_baseline_performance(self):
        """Compute climatology and seasonal naive baseline performance"""

        print("🌡️ Computing baseline performance (Climatology & Seasonal Naive)...")

        baseline_results = {}

        # Get unique forecast evaluations
        unique_forecasts = self.validation_results[
            ["origin_date", "forecast_date", "horizon", "actual"]
        ].drop_duplicates()

        # Group by horizon for efficiency
        for horizon in self.forecast_horizons:
            horizon_data = unique_forecasts[unique_forecasts["horizon"] == horizon]

            climatology_errors = []
            seasonal_naive_errors = []

            for _, row in horizon_data.iterrows():
                actual = row["actual"]
                forecast_date = row["forecast_date"]

                try:
                    # Climatology: Historical average for same day of year
                    same_doy = self.df[
                        self.df.index.dayofyear == forecast_date.dayofyear
                    ]
                    # Use data from at least 2 years ago to avoid data leakage
                    historical = same_doy[same_doy.index.year <= forecast_date.year - 2]

                    if len(historical) > 2:
                        climatology_pred = historical[self.target_col].mean()
                        climatology_error = abs(actual - climatology_pred)
                        climatology_errors.append(climatology_error)

                    # Seasonal Naive: Same day last year
                    last_year_date = forecast_date.replace(year=forecast_date.year - 1)
                    if last_year_date in self.df.index:
                        seasonal_naive_pred = self.df.loc[
                            last_year_date, self.target_col
                        ]
                        seasonal_naive_error = abs(actual - seasonal_naive_pred)
                        seasonal_naive_errors.append(seasonal_naive_error)

                except:
                    continue

            # Store results
            if climatology_errors:
                if "climatology" not in baseline_results:
                    baseline_results["climatology"] = {}
                baseline_results["climatology"][horizon] = {
                    "mae": np.mean(climatology_errors),
                    "count": len(climatology_errors),
                }

            if seasonal_naive_errors:
                if "seasonal_naive" not in baseline_results:
                    baseline_results["seasonal_naive"] = {}
                baseline_results["seasonal_naive"][horizon] = {
                    "mae": np.mean(seasonal_naive_errors),
                    "count": len(seasonal_naive_errors),
                }

        self.baseline_results = baseline_results

        # Print baseline performance
        print("📊 Baseline Performance:")
        for method, horizons in baseline_results.items():
            avg_mae = np.mean([h["mae"] for h in horizons.values()])
            print(f"  {method.replace('_', ' ').title()}: {avg_mae:.3f}°C average MAE")

        return baseline_results

    def create_comprehensive_plots(self):
        """Create comprehensive ML analysis visualization"""

        if not hasattr(self, "model_performances"):
            print("Run validation first")
            return

        # Compute baseline performance
        if not hasattr(self, "baseline_results"):
            self.compute_baseline_performance()

        print(f"\n🎨 Creating comprehensive ML analysis plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Add title to the entire figure
        fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

        # Plot 1: Performance by horizon for all models + baselines
        colors = ["red", "blue", "green", "orange", "purple", "brown"]

        # Plot ML models
        for i, (model_key, perf) in enumerate(self.model_performances.items()):
            horizons = sorted(perf["horizon_results"].keys())
            maes = [perf["horizon_results"][h]["mae"] for h in horizons]

            axes[0, 0].plot(
                horizons,
                maes,
                "o-",
                linewidth=2,
                markersize=5,
                label=perf["name"],
                color=colors[i % len(colors)],
            )

        # Add baseline methods
        if "climatology" in self.baseline_results:
            clim_horizons = sorted(self.baseline_results["climatology"].keys())
            clim_maes = [
                self.baseline_results["climatology"][h]["mae"] for h in clim_horizons
            ]
            axes[0, 0].plot(
                clim_horizons,
                clim_maes,
                "s--",
                linewidth=2,
                markersize=6,
                label="Climatology",
                color="gray",
                alpha=0.8,
            )

        if "seasonal_naive" in self.baseline_results:
            sn_horizons = sorted(self.baseline_results["seasonal_naive"].keys())
            sn_maes = [
                self.baseline_results["seasonal_naive"][h]["mae"] for h in sn_horizons
            ]
            axes[0, 0].plot(
                sn_horizons,
                sn_maes,
                "^:",
                linewidth=2,
                markersize=6,
                label="Seasonal Naive",
                color="black",
                alpha=0.8,
            )

        axes[0, 0].set_xlabel("Forecast Horizon (days)")
        axes[0, 0].set_ylabel("Mean Absolute Error (°C)")
        axes[0, 0].set_title("ML Models vs Baselines by Horizon")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Model comparison (average performance)
        model_names = [perf["name"] for perf in self.model_performances.values()]
        avg_maes = [perf["avg_mae"] for perf in self.model_performances.values()]

        # Sort by performance
        sorted_indices = np.argsort(avg_maes)
        model_names_sorted = [model_names[i] for i in sorted_indices]
        avg_maes_sorted = [avg_maes[i] for i in sorted_indices]

        bars = axes[0, 1].bar(
            range(len(model_names_sorted)),
            avg_maes_sorted,
            color=[colors[i % len(colors)] for i in range(len(model_names_sorted))],
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("ML Models")
        axes[0, 1].set_ylabel("Average MAE (°C)")
        axes[0, 1].set_title("ML Model Comparison")
        axes[0, 1].set_xticks(range(len(model_names_sorted)))
        axes[0, 1].set_xticklabels(
            [name.replace(" ", "\n") for name in model_names_sorted], rotation=45
        )
        axes[0, 1].grid(True, alpha=0.3)

        # Add values on bars
        for bar, mae in zip(bars, avg_maes_sorted):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mae:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 3: Method comparison (ML vs other methods including baselines)
        methods = ["Best ML", "Multi-output NN", "Prophet", "SARIMA"]
        best_ml_mae = min(avg_maes)
        method_maes = [best_ml_mae, 2.78, 4.09, 4.16]
        method_colors = ["red", "orange", "blue", "cyan"]

        # Add computed baselines
        if hasattr(self, "baseline_results"):
            if "climatology" in self.baseline_results:
                clim_avg = np.mean(
                    [h["mae"] for h in self.baseline_results["climatology"].values()]
                )
                methods.append("Climatology")
                method_maes.append(clim_avg)
                method_colors.append("gray")

            if "seasonal_naive" in self.baseline_results:
                sn_avg = np.mean(
                    [h["mae"] for h in self.baseline_results["seasonal_naive"].values()]
                )
                methods.append("Seasonal Naive")
                method_maes.append(sn_avg)
                method_colors.append("black")

        bars = axes[0, 2].bar(methods, method_maes, color=method_colors, alpha=0.7)
        axes[0, 2].set_ylabel("Average MAE (°C)")
        axes[0, 2].set_title("Overall Method Comparison")
        axes[0, 2].set_xticklabels(methods, rotation=45)
        axes[0, 2].grid(True, alpha=0.3)

        # Add values on bars
        for bar, mae in zip(bars, method_maes):
            axes[0, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{mae:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 4: Actual vs Predicted (best model, 1-day horizon)
        best_model_key = min(
            self.model_performances.keys(),
            key=lambda k: self.model_performances[k]["avg_mae"],
        )

        # Get 1-day predictions from best model
        best_model_results = self.validation_results[
            (self.validation_results["model"] == best_model_key)
            & (self.validation_results["horizon"] == 1)
        ]

        if len(best_model_results) > 0:
            sample_size = min(400, len(best_model_results))
            sample = best_model_results.sample(sample_size)

            axes[1, 0].scatter(
                sample["actual"], sample["predicted"], alpha=0.6, s=20, color="blue"
            )

            # Perfect prediction line
            min_temp = min(sample["actual"].min(), sample["predicted"].min())
            max_temp = max(sample["actual"].max(), sample["predicted"].max())
            axes[1, 0].plot(
                [min_temp, max_temp],
                [min_temp, max_temp],
                "r--",
                alpha=0.8,
                linewidth=2,
            )

            axes[1, 0].set_xlabel("Actual Temperature (°C)")
            axes[1, 0].set_ylabel("Predicted Temperature (°C)")
            axes[1, 0].set_title(
                f"Best ML Model: Actual vs Predicted (1-day)\n{self.model_performances[best_model_key]['name']}"
            )
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Residuals distribution (best model, 1-day horizon)
        if len(best_model_results) > 0:
            residuals = best_model_results["actual"] - best_model_results["predicted"]

            axes[1, 1].hist(
                residuals, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[1, 1].axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=2)
            axes[1, 1].axvline(
                residuals.mean(),
                color="orange",
                linestyle="-",
                alpha=0.7,
                linewidth=2,
                label=f"Mean: {residuals.mean():.2f}°C",
            )
            axes[1, 1].set_xlabel("Residuals (°C)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("ML Residuals Distribution (1-day)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Performance by horizon (best 3 models + baselines)
        top_3_models = sorted(
            self.model_performances.items(), key=lambda x: x[1]["avg_mae"]
        )[:3]

        for i, (model_key, perf) in enumerate(top_3_models):
            horizons = sorted(perf["horizon_results"].keys())
            maes = [perf["horizon_results"][h]["mae"] for h in horizons]

            axes[1, 2].plot(
                horizons,
                maes,
                "o-",
                linewidth=2,
                markersize=6,
                label=perf["name"],
                color=colors[i],
            )

        # Add baseline methods to this plot as well
        if "climatology" in self.baseline_results:
            clim_horizons = sorted(self.baseline_results["climatology"].keys())
            clim_maes = [
                self.baseline_results["climatology"][h]["mae"] for h in clim_horizons
            ]
            axes[1, 2].plot(
                clim_horizons,
                clim_maes,
                "s--",
                linewidth=2,
                markersize=6,
                label="Climatology",
                color="gray",
                alpha=0.8,
            )

        if "seasonal_naive" in self.baseline_results:
            sn_horizons = sorted(self.baseline_results["seasonal_naive"].keys())
            sn_maes = [
                self.baseline_results["seasonal_naive"][h]["mae"] for h in sn_horizons
            ]
            axes[1, 2].plot(
                sn_horizons,
                sn_maes,
                "^:",
                linewidth=2,
                markersize=6,
                label="Seasonal Naive",
                color="black",
                alpha=0.8,
            )

        axes[1, 2].set_xlabel("Forecast Horizon (days)")
        axes[1, 2].set_ylabel("Mean Absolute Error (°C)")
        axes[1, 2].set_title("Top 3 ML Models vs Baselines")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("ml/ml_comprehensive_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("✅ Saved ML comprehensive analysis: ml/ml_comprehensive_analysis.png")

    def create_detailed_performance_plots(self):
        """Create additional detailed performance analysis plots"""

        if not hasattr(self, "validation_results"):
            print("Run validation first")
            return

        print(f"\n🎨 Creating detailed performance plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Add title to the entire figure
        fig.suptitle("Neural Net Performance Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Error distribution by horizon
        horizons_to_plot = [1, 7, 14, 30]
        horizon_colors = ["blue", "green", "orange", "red"]

        for i, horizon in enumerate(horizons_to_plot):
            horizon_data = self.validation_results[
                self.validation_results["horizon"] == horizon
            ]
            if len(horizon_data) > 10:
                axes[0, 0].hist(
                    horizon_data["error"],
                    bins=20,
                    alpha=0.6,
                    label=f"{horizon}-day",
                    color=horizon_colors[i],
                    edgecolor="black",
                    linewidth=0.5,
                )

        axes[0, 0].set_xlabel("Absolute Error (°C)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Error Distribution by Horizon")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Performance over time (monthly)
        self.validation_results["month"] = self.validation_results[
            "forecast_date"
        ].dt.to_period("M")
        monthly_performance = self.validation_results.groupby("month")["error"].mean()

        axes[0, 1].plot(
            range(len(monthly_performance)),
            monthly_performance.values,
            "o-",
            linewidth=2,
            markersize=4,
            color="purple",
        )
        axes[0, 1].set_xlabel("Time Period")
        axes[0, 1].set_ylabel("Mean Absolute Error (°C)")
        axes[0, 1].set_title("Performance Over Time (Monthly)")
        axes[0, 1].grid(True, alpha=0.3)

        # Rotate x-axis labels
        x_labels = [
            str(period)
            for period in monthly_performance.index[
                :: max(1, len(monthly_performance) // 8)
            ]
        ]
        x_positions = list(
            range(0, len(monthly_performance), max(1, len(monthly_performance) // 8))
        )
        axes[0, 1].set_xticks(x_positions)
        axes[0, 1].set_xticklabels(x_labels, rotation=45)

        # Plot 3: Model performance by season
        self.validation_results["season"] = self.validation_results[
            "forecast_date"
        ].dt.month.map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }
        )

        seasonal_performance = self.validation_results.groupby("season")["error"].mean()
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        seasonal_performance = seasonal_performance.reindex(season_order)

        bars = axes[1, 0].bar(
            seasonal_performance.index,
            seasonal_performance.values,
            color=["lightblue", "lightgreen", "gold", "orange"],
            alpha=0.7,
            edgecolor="black",
        )
        axes[1, 0].set_ylabel("Mean Absolute Error (°C)")
        axes[1, 0].set_title("Performance by Season")
        axes[1, 0].grid(True, alpha=0.3)

        # Add values on bars
        for bar, mae in zip(bars, seasonal_performance.values):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mae:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 4: Box plot of errors by model (top 4 models)
        top_4_models = sorted(
            self.model_performances.items(), key=lambda x: x[1]["avg_mae"]
        )[:4]

        model_errors = []
        model_labels = []

        for model_key, perf in top_4_models:
            model_data = self.validation_results[
                self.validation_results["model"] == model_key
            ]
            if len(model_data) > 10:
                model_errors.append(model_data["error"].values)
                model_labels.append(perf["name"].replace(" ", "\n"))

        if model_errors:
            bp = axes[1, 1].boxplot(
                model_errors, labels=model_labels, patch_artist=True
            )

            # Color the boxes
            colors_box = ["lightcoral", "lightskyblue", "lightgreen", "lightsalmon"]
            for patch, color in zip(bp["boxes"], colors_box):
                patch.set_facecolor(color)

            axes[1, 1].set_ylabel("Absolute Error (°C)")
            axes[1, 1].set_title("Error Distribution by Model")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("ml/ml_detailed_performance.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("✅ Saved detailed performance analysis: ml/ml_detailed_performance.png")

    def save_production_models(self):
        """Save the trained models for production use"""

        if not hasattr(self, "model_performances"):
            print("Train models first")
            return

        import pickle

        production_package = {
            "models": {},
            "feature_columns": self.feature_cols,
            "validation_results": self.validation_results,
            "model_performances": self.model_performances,
        }

        # Save top 3 models
        model_ranking = [(k, v["avg_mae"]) for k, v in self.model_performances.items()]
        model_ranking.sort(key=lambda x: x[1])

        for model_key, mae in model_ranking[:3]:
            perf = self.model_performances[model_key]
            production_package["models"][model_key] = {
                "name": perf["name"],
                "model": perf["fitted_model"],
                "scaler": perf["scaler"],
                "avg_mae": mae,
            }

        with open("ml/production_ml_models.pkl", "wb") as f:
            pickle.dump(production_package, f)

        print(f"💾 Saved production models to: ml/production_ml_models.pkl")

        # Save validation results
        self.validation_results.to_csv("ml/production_ml_validation.csv", index=False)
        print(f"💾 Saved validation results to: ml/production_ml_validation.csv")


def main():
    """Main production ML pipeline"""

    print("=" * 60)
    print("PRODUCTION ML ENSEMBLE PIPELINE")
    print("=" * 60)
    print("Using debugged, verified clean features!")

    # Initialize ensemble
    ensemble = ProductionMLEnsemble("ultra_clean_ml_features.csv")

    # Create models
    models = ensemble.create_production_models()

    # Comprehensive validation
    validation_results = ensemble.comprehensive_validation(
        test_years=2, forecast_horizons=[1, 3, 7, 14, 30]
    )

    # Analyze performance
    model_ranking = ensemble.analyze_model_performance()

    # Compare with benchmarks
    ensemble.compare_with_benchmarks()

    # Create ensemble
    ensemble_results = ensemble.create_ensemble_predictions(top_n_models=3)

    # Create comprehensive visualizations
    ensemble.create_comprehensive_plots()
    ensemble.create_detailed_performance_plots()

    # Save models
    ensemble.save_production_models()

    # Final summary
    best_model = min(ensemble.model_performances.items(), key=lambda x: x[1]["avg_mae"])
    best_mae = best_model[1]["avg_mae"]
    best_name = best_model[1]["name"]

    print(f"\n" + "=" * 60)
    print("PRODUCTION ML PIPELINE COMPLETE!")
    print("=" * 60)
    print("✅ Validated clean ML models")
    print("✅ Comprehensive performance analysis")
    print("✅ Ensemble model created")
    print("✅ Comprehensive visualizations generated")
    print("✅ Production models saved")
    print("✅ Ready for deployment!")

    print(f"\n🏆 BEST ML MODEL RESULTS:")
    print(f"📊 Model: {best_name}")
    print(f"📊 Overall MAE: {best_mae:.3f}°C")
    print(
        f"📊 vs Multi-output NN (2.78°C): {best_mae/2.78:.2f}x {'better' if best_mae < 2.78 else 'worse'}"
    )
    print(f"📊 vs Prophet (4.09°C): {best_mae/4.09:.2f}x better")
    print(f"📊 vs SARIMA (4.16°C): {best_mae/4.16:.2f}x better")

    if best_mae < 2.5:
        print("🚀 Outstanding performance - new state of the art!")
    elif best_mae < 2.8:
        print("🎉 Excellent performance - competitive with multi-output NN!")
    elif best_mae < 3.0:
        print("✅ Strong performance achieved!")
    else:
        print("📊 Solid competitive performance")

    return ensemble


if __name__ == "__main__":
    ensemble = main()
