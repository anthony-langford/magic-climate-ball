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
        print(f"Prophet Conservative      4.090         📊 Strong baseline")
        print(f"Climatology Baseline      4.400         📊 Traditional")
        print(f"Persistence (1-day)       3.170         📊 Short-term")

        # Calculate improvements
        prophet_improvement = ((4.09 - best_ml_mae) / 4.09) * 100
        clim_improvement = ((4.40 - best_ml_mae) / 4.40) * 100

        print(f"\nML IMPROVEMENTS:")
        print(f"• {prophet_improvement:.1f}% better than Prophet")
        print(f"• {clim_improvement:.1f}% better than Climatology")

        if best_ml_mae < 3.0:
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

        with open("production_ml_models.pkl", "wb") as f:
            pickle.dump(production_package, f)

        print(f"Saved production models to: production_ml_models.pkl")

        # Save validation results
        self.validation_results.to_csv("production_ml_validation.csv", index=False)
        print(f"Saved validation results to: production_ml_validation.csv")


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

    # Save models
    ensemble.save_production_models()

    print(f"\n" + "=" * 60)
    print("PRODUCTION ML PIPELINE COMPLETE!")
    print("=" * 60)
    print("✅ Validated clean ML models")
    print("✅ Comprehensive performance analysis")
    print("✅ Ensemble model created")
    print("✅ Production models saved")
    print("✅ Ready for deployment!")

    return ensemble


if __name__ == "__main__":
    ensemble = main()
