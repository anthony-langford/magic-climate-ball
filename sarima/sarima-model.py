import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import os

warnings.filterwarnings("ignore")


class SeasonalSARIMAForecaster:
    """SARIMA with proper seasonal components to beat climatology"""

    def __init__(self, data_path="temagami_features.csv"):
        """Load data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.temp_series = self.df["t_mean"].copy()

        print(f"🌡️ Seasonal SARIMA Forecaster")
        print(f"📊 Data: {len(self.temp_series)} observations")
        print(
            f"📅 Range: {self.temp_series.index.min().date()} to {self.temp_series.index.max().date()}"
        )

        os.makedirs("sarima", exist_ok=True)

    def seasonal_model_search(self):
        """Search for models with proper seasonal components"""
        print(f"\n🌀 Seasonal Model Search")
        print("=" * 40)

        # Models with annual seasonality (period=365) - computationally expensive but necessary
        seasonal_models = [
            # Simple seasonal models
            {
                "order": (0, 1, 1),
                "seasonal_order": (1, 0, 0, 365),
                "name": "Seasonal AR(1)",
            },
            {
                "order": (0, 1, 1),
                "seasonal_order": (0, 0, 1, 365),
                "name": "Seasonal MA(1)",
            },
            {
                "order": (1, 1, 1),
                "seasonal_order": (1, 0, 0, 365),
                "name": "Seasonal ARIMA(1,1,1)",
            },
            # The best non-seasonal model for comparison
            {
                "order": (0, 1, 2),
                "seasonal_order": (0, 0, 0, 0),
                "name": "MA(2) no seasonal",
            },
            # Compromise: Weekly seasonality (less expensive)
            {
                "order": (0, 1, 2),
                "seasonal_order": (1, 0, 0, 7),
                "name": "MA(2) + Weekly AR",
            },
            {
                "order": (1, 1, 1),
                "seasonal_order": (1, 0, 1, 7),
                "name": "ARIMA + Weekly",
            },
        ]

        # Use smaller dataset for seasonal model fitting (computational constraint)
        train_size = min(1200, len(self.temp_series) - 365)  # Last 3+ years
        train_data = self.temp_series.iloc[
            -train_size - 365 : -365
        ]  # Leave 1 year for testing

        print(f"📊 Using {len(train_data)} observations for seasonal model search")
        print(
            f"⚠️ Annual seasonality is computationally expensive - this may take time..."
        )

        results = []
        best_model = None
        best_score = np.inf

        for i, config in enumerate(seasonal_models):
            print(f"\n[{i+1}/{len(seasonal_models)}] Testing {config['name']}")
            print(f"    SARIMA{config['order']} x {config['seasonal_order']}")

            try:
                # For annual seasonality, use concentrated scale and simpler initialization
                use_concentrated = config["seasonal_order"][3] == 365

                model = SARIMAX(
                    train_data,
                    order=config["order"],
                    seasonal_order=config["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=use_concentrated,  # More efficient for seasonal models
                    initialization="approximate_diffuse",
                )

                print(f"    📊 Fitting model...")
                fitted = model.fit(
                    disp=False, maxiter=100 if use_concentrated else 200, method="lbfgs"
                )

                if fitted.mle_retvals["converged"]:
                    aic = fitted.aic
                    print(f"    ✅ Converged: AIC = {aic:.1f}")

                    # Quick forecast test
                    try:
                        test_forecast = fitted.forecast(steps=10)
                        if not np.isnan(test_forecast).any():
                            print(f"    ✅ Forecasting works")

                            # Simple validation score (AIC + forecast penalty)
                            score = aic + (
                                1000 if np.any(np.abs(test_forecast) > 50) else 0
                            )

                            results.append(
                                {
                                    "config": config,
                                    "aic": aic,
                                    "score": score,
                                    "fitted_model": fitted,
                                }
                            )

                            if score < best_score:
                                best_score = score
                                best_model = config
                                print(f"    🏆 New best model!")

                        else:
                            print(f"    ❌ Forecasting returns NaN")
                    except Exception as e:
                        print(f"    ❌ Forecasting failed: {e}")
                else:
                    print(f"    ❌ Failed to converge")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue

        if best_model is None:
            print("\n❌ No seasonal models worked! Using MA(2) fallback")
            best_model = {
                "order": (0, 1, 2),
                "seasonal_order": (0, 0, 0, 0),
                "name": "MA(2) fallback",
            }

        print(f"\n🏆 Selected Model: {best_model['name']}")
        print(f"📊 SARIMA{best_model['order']} x {best_model['seasonal_order']}")

        self.best_model = best_model
        return best_model

    def seasonal_validation(self, test_years=2):
        """Validation with focus on seasonal performance"""
        print(f"\n🚀 Seasonal Validation")
        print("=" * 30)

        # Split data - use timedelta for fractional years
        days_back = int(test_years * 365.25)  # Account for leap years
        split_date = self.temp_series.index.max() - timedelta(days=days_back)
        train_data = self.temp_series[self.temp_series.index <= split_date]
        test_data = self.temp_series[self.temp_series.index > split_date]

        print(
            f"📊 Training: {len(train_data)} obs ({train_data.index.min().date()} to {train_data.index.max().date()})"
        )
        print(
            f"📊 Testing: {len(test_data)} obs ({test_data.index.min().date()} to {test_data.index.max().date()})"
        )

        if not hasattr(self, "best_model"):
            self.seasonal_model_search()

        # Investigate why seasonal models have worse AIC
        print(f"\n🔍 Analyzing Model Selection:")
        print(
            f"Selected: {self.best_model['name']} - SARIMA{self.best_model['order']} x {self.best_model['seasonal_order']}"
        )

        # Check if we actually selected a seasonal model
        is_seasonal = any(x != 0 for x in self.best_model["seasonal_order"][:3])
        seasonal_period = (
            self.best_model["seasonal_order"][3]
            if len(self.best_model["seasonal_order"]) > 3
            else 0
        )

        if not is_seasonal:
            print("⚠️  WARNING: Selected model has NO seasonal component!")
            print("⚠️  This explains why climatology beats SARIMA at 7+ days")
            print(
                "⚠️  Annual seasonality may be too complex for this SARIMA implementation"
            )
        else:
            print(f"✅ Selected model has seasonality (period={seasonal_period})")

        # Test every 21 days (3 weeks) for efficiency but good seasonal coverage
        test_points = test_data.index[::21]
        print(f"🎯 Testing at {len(test_points)} points (every 3 weeks)")

        results = []
        max_horizon = 30

        for i, test_date in enumerate(test_points):
            if i % 5 == 0:
                print(f"📈 Progress: {i+1}/{len(test_points)}")

            try:
                # Get training data - for seasonal models, use more data
                current_train = self.temp_series[
                    self.temp_series.index <= test_date - timedelta(days=1)
                ]

                # Use appropriate amount of data based on seasonality
                if seasonal_period == 365:
                    # Annual seasonality - need at least 2-3 years
                    min_data = 365 * 2
                    max_data = 365 * 4
                elif seasonal_period == 7:
                    # Weekly seasonality - need less data
                    min_data = 365
                    max_data = 365 * 3
                else:
                    # No seasonality
                    min_data = 500
                    max_data = 1500

                if len(current_train) > max_data:
                    current_train = current_train.iloc[-max_data:]
                elif len(current_train) < min_data:
                    continue  # Not enough data

                # Fit model
                model = SARIMAX(
                    current_train,
                    order=self.best_model["order"],
                    seasonal_order=self.best_model["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=seasonal_period == 365,
                    initialization="approximate_diffuse",
                )

                fitted = model.fit(disp=False, maxiter=100)

                if not fitted.mle_retvals.get("converged", False):
                    continue

                # Generate forecasts
                forecasts = fitted.forecast(steps=max_horizon)

                # Collect results
                for h in range(1, max_horizon + 1):
                    forecast_date = test_date + timedelta(days=h)

                    if forecast_date in test_data.index:
                        actual = test_data.loc[forecast_date]
                        predicted = forecasts.iloc[h - 1]

                        if not np.isnan(predicted) and -50 <= predicted <= 50:
                            results.append(
                                {
                                    "horizon": h,
                                    "actual": actual,
                                    "predicted": predicted,
                                    "error": abs(actual - predicted),
                                    "forecast_date": forecast_date,
                                }
                            )

            except Exception as e:
                continue

        self.results = pd.DataFrame(results)

        if len(self.results) > 0:
            overall_mae = self.results["error"].mean()
            print(f"\n✅ Seasonal Validation Complete!")
            print(f"📊 Generated {len(self.results)} forecasts")
            print(f"🎯 Overall MAE: {overall_mae:.3f}°C")

            # Performance by horizon
            print(f"\nPerformance by horizon:")
            for h in [1, 3, 7, 14, 21, 30]:
                h_data = self.results[self.results["horizon"] == h]
                if len(h_data) > 3:
                    mae = h_data["error"].mean()
                    print(f"  {h:2d} days: {mae:.2f}°C (n={len(h_data)})")

            return self.results
        else:
            print("❌ No validation results generated")
            return pd.DataFrame()

    def compare_with_climatology(self):
        """Compare specifically with climatology (the main competitor)"""
        if not hasattr(self, "results") or len(self.results) == 0:
            return

        print(f"\n🌡️ Seasonal SARIMA vs Climatology")
        print("=" * 40)

        # Compute climatology baseline
        climatology_results = []

        for _, row in self.results.iterrows():
            actual = row["actual"]
            forecast_date = row["forecast_date"]

            # Climatology: historical average for same day of year
            try:
                same_doy = self.temp_series[
                    self.temp_series.index.dayofyear == forecast_date.dayofyear
                ]
                historical = same_doy[same_doy.index.year < forecast_date.year - 1]
                if len(historical) > 2:
                    climatology_pred = historical.mean()
                    climatology_results.append(
                        {
                            "horizon": row["horizon"],
                            "error": abs(actual - climatology_pred),
                        }
                    )
            except:
                pass

        if len(climatology_results) > 0:
            climatology_df = pd.DataFrame(climatology_results)
            climatology_mae = climatology_df.groupby("horizon")["error"].mean()
            sarima_mae = self.results.groupby("horizon")["error"].mean()

            print(
                f"{'Horizon':<8} {'SARIMA':<8} {'Climatology':<12} {'Improvement':<12} {'Winner'}"
            )
            print("-" * 50)

            wins = 0
            total = 0

            for h in [1, 3, 7, 14, 21, 30]:
                if h in sarima_mae.index and h in climatology_mae.index:
                    s_mae = sarima_mae[h]
                    c_mae = climatology_mae[h]
                    improvement = ((c_mae - s_mae) / c_mae) * 100

                    winner = "SARIMA ✅" if improvement > 0 else "Climatology ❌"
                    if improvement > 0:
                        wins += 1
                    total += 1

                    print(
                        f"{h:<8} {s_mae:<8.2f} {c_mae:<12.2f} {improvement:<+11.1f}% {winner}"
                    )

            print(f"\nSeasonal SARIMA wins: {wins}/{total} horizons")

            if wins >= total * 0.7:
                print("🎉 Seasonal SARIMA beats climatology!")
            elif wins >= total * 0.5:
                print("📊 Mixed results - some improvement")
            else:
                print("⚠️ Still losing to climatology")

    def hybrid_approach_test(self):
        """Test hybrid SARIMA + Climatology approach"""
        if not hasattr(self, "results") or len(self.results) == 0:
            print("❌ No SARIMA results for hybrid approach")
            return None

        print(f"\n🔄 Testing Hybrid SARIMA + Climatology")
        print("=" * 45)

        # Define horizon cutoff
        sarima_horizons = [1, 2, 3]  # SARIMA is good for short-term
        climatology_horizons = [7, 14, 21, 30]  # Climatology is good for long-term

        hybrid_results = []

        # Get unique forecast origins from SARIMA results
        forecast_origins = self.results["forecast_date"].dt.date.unique()

        for forecast_date in forecast_origins:
            try:
                # Get actual temperature for this date
                if pd.Timestamp(forecast_date) in self.temp_series.index:
                    actual = self.temp_series.loc[pd.Timestamp(forecast_date)]

                    # For short horizons: use SARIMA results
                    for h in sarima_horizons:
                        sarima_result = self.results[
                            (self.results["forecast_date"].dt.date == forecast_date)
                            & (self.results["horizon"] == h)
                        ]
                        if len(sarima_result) > 0:
                            hybrid_results.append(
                                {
                                    "horizon": h,
                                    "actual": actual,
                                    "predicted": sarima_result.iloc[0]["predicted"],
                                    "error": sarima_result.iloc[0]["error"],
                                    "method": "SARIMA",
                                    "forecast_date": forecast_date,
                                }
                            )

                    # For long horizons: use climatology
                    for h in climatology_horizons:
                        # Calculate climatology prediction
                        target_date = pd.Timestamp(forecast_date)
                        same_doy = self.temp_series[
                            self.temp_series.index.dayofyear == target_date.dayofyear
                        ]
                        historical = same_doy[
                            same_doy.index.year < target_date.year - 1
                        ]

                        if len(historical) > 2:
                            climatology_pred = historical.mean()
                            error = abs(actual - climatology_pred)

                            hybrid_results.append(
                                {
                                    "horizon": h,
                                    "actual": actual,
                                    "predicted": climatology_pred,
                                    "error": error,
                                    "method": "Climatology",
                                    "forecast_date": forecast_date,
                                }
                            )
            except:
                continue

        if len(hybrid_results) > 0:
            hybrid_df = pd.DataFrame(hybrid_results)
            hybrid_mae = hybrid_df["error"].mean()

            print(f"📊 Hybrid Results:")
            print(f"Overall MAE: {hybrid_mae:.3f}°C")

            # Performance by horizon
            print(f"\nHybrid performance by horizon:")
            for h in [1, 2, 3, 7, 14, 21, 30]:
                h_data = hybrid_df[hybrid_df["horizon"] == h]
                if len(h_data) > 0:
                    mae = h_data["error"].mean()
                    method = h_data.iloc[0]["method"]
                    print(f"  {h:2d} days: {mae:.2f}°C (n={len(h_data)}) - {method}")

            # Compare to pure SARIMA
            sarima_mae = self.results["error"].mean()
            improvement = ((sarima_mae - hybrid_mae) / sarima_mae) * 100

            print(f"\n🔍 Hybrid vs Pure SARIMA:")
            print(f"Pure SARIMA: {sarima_mae:.3f}°C")
            print(f"Hybrid: {hybrid_mae:.3f}°C")
            print(f"Improvement: {improvement:+.1f}%")

            self.hybrid_results = hybrid_df
            return hybrid_df
        else:
            print("❌ No hybrid results generated")
            return None

    def create_seasonal_plots(self):
        """Create plots showing seasonal performance"""
        if not hasattr(self, "results") or len(self.results) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. MAE by horizon
        sarima_perf = self.results.groupby("horizon")["error"].mean()
        horizons = [h for h in sarima_perf.index if h <= 30]

        axes[0, 0].plot(
            horizons,
            [sarima_perf[h] for h in horizons],
            "o-",
            linewidth=2,
            markersize=5,
            color="red",
            label="Seasonal SARIMA",
        )

        # Add climatology comparison if available
        climatology_results = self.compare_with_climatology()
        if climatology_results:
            clim_df = pd.DataFrame(climatology_results)
            clim_perf = clim_df.groupby("horizon")["error"].mean()
            axes[0, 0].plot(
                clim_perf.index,
                clim_perf.values,
                "s--",
                linewidth=2,
                label="Climatology",
                alpha=0.8,
                color="blue",
            )

        axes[0, 0].set_xlabel("Forecast Horizon (days)")
        axes[0, 0].set_ylabel("Mean Absolute Error (°C)")
        axes[0, 0].set_title(f'Seasonal SARIMA Performance\n{self.best_model["name"]}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Seasonal performance (by month)
        monthly_results = self.results.copy()
        monthly_results["month"] = monthly_results["forecast_date"].dt.month
        monthly_mae = monthly_results.groupby("month")["error"].mean()

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        axes[0, 1].plot(
            monthly_mae.index,
            monthly_mae.values,
            "o-",
            linewidth=2,
            markersize=5,
            color="green",
        )
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Mean Absolute Error (°C)")
        axes[0, 1].set_title("Performance by Season")
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(month_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Actual vs Predicted
        sample = self.results.sample(min(400, len(self.results)))
        axes[1, 0].scatter(sample["actual"], sample["predicted"], alpha=0.6, s=15)

        temp_range = [sample["actual"].min(), sample["actual"].max()]
        axes[1, 0].plot(temp_range, temp_range, "r--", alpha=0.8)
        axes[1, 0].set_xlabel("Actual Temperature (°C)")
        axes[1, 0].set_ylabel("Predicted Temperature (°C)")
        axes[1, 0].set_title("Actual vs Predicted")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Error by horizon (box plot style)
        key_horizons = [1, 7, 14, 30]
        horizon_errors = []
        horizon_labels = []

        for h in key_horizons:
            h_data = self.results[self.results["horizon"] == h]
            if len(h_data) > 10:
                horizon_errors.append(h_data["error"].values)
                horizon_labels.append(f"{h}d")

        if horizon_errors:
            axes[1, 1].boxplot(horizon_errors, labels=horizon_labels)
            axes[1, 1].set_xlabel("Forecast Horizon")
            axes[1, 1].set_ylabel("Absolute Error (°C)")
            axes[1, 1].set_title("Error Distribution by Horizon")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "sarima/seasonal_sarima_performance.png", dpi=150, bbox_inches="tight"
        )
        plt.show()


def main():
    """Main function for seasonal SARIMA"""
    print("🌀 SEASONAL SARIMA FOR TEMPERATURE FORECASTING")
    print("=" * 55)

    try:
        forecaster = SeasonalSARIMAForecaster("temagami_features.csv")

        # Search for best seasonal model
        best_model = forecaster.seasonal_model_search()

        # Seasonal validation
        results = forecaster.seasonal_validation(test_years=2)  # Fixed to use integer

        if len(results) > 0:
            # Compare with climatology
            forecaster.compare_with_climatology()

            # Test hybrid approach
            hybrid_results = forecaster.hybrid_approach_test()

            # Create plots
            forecaster.create_seasonal_plots()

            # Final summary
            overall_mae = results["error"].mean()
            print(f"\n🎯 SEASONAL SARIMA RESULTS")
            print("=" * 35)
            print(f"📊 Model: {best_model['name']}")
            print(f"📊 Overall MAE: {overall_mae:.3f}°C")

            # Compare to previous best
            print(f"📊 vs Previous SARIMA (5.39°C): {overall_mae/5.39:.2f}x better")
            print(
                f"📊 vs Neural Net (2.78°C): {overall_mae/2.78:.2f}x {'better' if overall_mae < 2.78 else 'worse'}"
            )

            # Show hybrid results if available
            if hybrid_results is not None:
                hybrid_mae = hybrid_results["error"].mean()
                print(f"📊 Hybrid approach: {hybrid_mae:.3f}°C")
                print(
                    f"📊 vs Neural Net (2.78°C): {hybrid_mae/2.78:.2f}x {'better' if hybrid_mae < 2.78 else 'worse'}"
                )

            # Check if we need a hybrid approach
            is_seasonal = any(x != 0 for x in best_model["seasonal_order"][:3])
            if not is_seasonal:
                print(f"\n💡 INSIGHT: ANNUAL SEASONALITY TOO COMPLEX FOR SARIMA")
                print(f"   • SARIMA excels at short-term patterns (1-3 days)")
                print(
                    f"   • Climatology captures long-term seasonal patterns (7+ days)"
                )
                print(f"   • Neural networks handle both patterns simultaneously")

            return forecaster, results
        else:
            print("❌ No results generated")
            return forecaster, None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    forecaster, results = main()
