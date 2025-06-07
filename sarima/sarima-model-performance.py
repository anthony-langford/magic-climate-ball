import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import itertools
import time
import pickle
import os
from multiprocessing import Pool, cpu_count
import gc

warnings.filterwarnings("ignore")


class RobustSARIMAForecaster:
    """Robust SARIMA forecaster with comprehensive grid search"""

    def __init__(self, data_path="temagami_features.csv"):
        """Load the feature-engineered data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.temp_series = self.df["t_mean"].copy()
        print(f"Loaded temperature series: {len(self.temp_series)} observations")
        print(
            f"Date range: {self.temp_series.index.min().date()} to {self.temp_series.index.max().date()}"
        )
        print(f"System: M2 Max with {os.cpu_count()} cores available")

    def check_stationarity(self, series, title="Series"):
        """Comprehensive stationarity analysis"""
        print(f"\nStationarity Analysis for {title}")
        print("-" * 50)

        # ADF test
        adf_result = adfuller(series.dropna(), autolag="AIC")
        print(f"ADF Statistic: {adf_result[0]:.6f}")
        print(f"p-value: {adf_result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in adf_result[4].items():
            print(f"  {key}: {value:.6f}")

        is_stationary = adf_result[1] <= 0.05
        print(f"Result: {'Stationary' if is_stationary else 'Non-stationary'}")

        return is_stationary, adf_result

    def determine_differencing(self):
        """Systematically determine optimal differencing"""
        print("\n" + "=" * 60)
        print("DIFFERENCING ANALYSIS")
        print("=" * 60)

        original = self.temp_series.dropna()

        # Test original series
        is_stat_orig, adf_orig = self.check_stationarity(original, "Original")

        # Test first difference
        diff1 = original.diff().dropna()
        is_stat_diff1, adf_diff1 = self.check_stationarity(diff1, "First Difference")

        # Test seasonal difference (365 days)
        seasonal_diff = original.diff(365).dropna()
        if len(seasonal_diff) > 100:  # Need enough data
            is_stat_seasonal, adf_seasonal = self.check_stationarity(
                seasonal_diff, "Seasonal Difference (365)"
            )
        else:
            is_stat_seasonal, adf_seasonal = False, None

        # Test both differences
        both_diff = original.diff().diff(365).dropna()
        if len(both_diff) > 100:
            is_stat_both, adf_both = self.check_stationarity(
                both_diff, "Both Differences"
            )
        else:
            is_stat_both, adf_both = False, None

        # Recommend differencing strategy
        print(f"\nDifferencing Recommendations:")
        if is_stat_orig:
            print("✓ No differencing needed (d=0, D=0)")
            return (0, 0)
        elif is_stat_diff1:
            print("✓ First differencing sufficient (d=1, D=0)")
            return (1, 0)
        elif is_stat_seasonal:
            print("✓ Seasonal differencing sufficient (d=0, D=1)")
            return (0, 1)
        elif is_stat_both:
            print("✓ Both differences needed (d=1, D=1)")
            return (1, 1)
        else:
            print("⚠ Default to first differencing (d=1, D=0)")
            return (1, 0)

    def fit_single_sarima(self, params):
        """Fit a single SARIMA model - designed for parallel processing"""
        train_data, order, seasonal_order, timeout_seconds = params

        start_time = time.time()

        try:
            # Create model
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization="approximate_diffuse",
                concentrate_scale=True,  # Can improve numerical stability
            )

            # Fit with timeout protection
            fitted_model = model.fit(
                disp=False,
                maxiter=50,  # Limit iterations to prevent hanging
                method="lbfgs",  # Often more stable than default
                optim_score="harvey",  # Alternative scoring
                low_memory=True,  # Memory optimization
            )

            # Check if fitting took too long
            if time.time() - start_time > timeout_seconds:
                return {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": np.inf,
                    "bic": np.inf,
                    "converged": False,
                    "error": "Timeout",
                }

            # Extract results
            result = {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "llf": fitted_model.llf,
                "converged": fitted_model.mle_retvals["converged"],
                "fit_time": time.time() - start_time,
                "error": None,
            }

            # Force garbage collection
            del fitted_model, model
            gc.collect()

            return result

        except Exception as e:
            return {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": np.inf,
                "bic": np.inf,
                "converged": False,
                "fit_time": time.time() - start_time,
                "error": str(e)[:100],
            }

    def comprehensive_grid_search(
        self,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        seasonal_period=365,
        n_jobs=None,
        save_progress=True,
    ):
        """Comprehensive SARIMA grid search optimized for M2 Max"""
        print(f"\n" + "=" * 60)
        print("COMPREHENSIVE SARIMA GRID SEARCH")
        print("=" * 60)

        # Determine differencing
        d_rec, D_rec = self.determine_differencing()

        # Set up training data (hold out last year for final validation)
        train_end = self.temp_series.index[-365]
        train_data = self.temp_series[self.temp_series.index <= train_end].dropna()

        print(f"\nGrid Search Configuration:")
        print(f"Training data: {len(train_data)} observations")
        print(
            f"Date range: {train_data.index.min().date()} to {train_data.index.max().date()}"
        )
        print(
            f"Parameter ranges: p=[0,{max_p}], q=[0,{max_q}], P=[0,{max_P}], Q=[0,{max_Q}]"
        )
        print(f"Differencing: d=[{d_rec-1},{d_rec},{d_rec+1}], D=[{D_rec},{D_rec+1}]")
        print(f"Seasonal period: {seasonal_period}")

        # Generate parameter combinations
        p_values = list(range(0, max_p + 1))
        d_values = [
            max(0, d_rec - 1),
            d_rec,
            min(2, d_rec + 1),
        ]  # Try around recommended
        q_values = list(range(0, max_q + 1))

        P_values = list(range(0, max_P + 1))
        D_values = [D_rec, min(2, D_rec + 1)]  # Try recommended and +1
        Q_values = list(range(0, max_Q + 1))

        # Remove duplicates and invalid combinations
        d_values = sorted(list(set(d_values)))
        D_values = sorted(list(set(D_values)))

        # Generate all combinations
        param_combinations = []
        for p, d, q in itertools.product(p_values, d_values, q_values):
            for P, D, Q in itertools.product(P_values, D_values, Q_values):
                # Skip if all parameters are zero
                if p + d + q + P + D + Q == 0:
                    continue

                order = (p, d, q)
                seasonal_order = (P, D, Q, seasonal_period)
                param_combinations.append(
                    (train_data, order, seasonal_order, 30)
                )  # 30 second timeout

        total_combinations = len(param_combinations)
        print(f"Total combinations to test: {total_combinations}")

        # Set up parallel processing
        if n_jobs is None:
            n_jobs = min(8, os.cpu_count())  # Don't overwhelm the system

        print(f"Using {n_jobs} parallel processes")

        # Progress tracking
        results = []
        batch_size = max(1, total_combinations // 20)  # 20 progress updates

        print(f"\nStarting grid search...")
        start_time = time.time()

        # Process in batches for progress tracking
        for i in range(0, total_combinations, batch_size):
            batch = param_combinations[i : i + batch_size]
            batch_start = time.time()

            # Parallel processing
            with Pool(processes=n_jobs) as pool:
                batch_results = pool.map(self.fit_single_sarima, batch)

            results.extend(batch_results)

            # Progress update
            completed = min(i + batch_size, total_combinations)
            elapsed = time.time() - start_time
            progress = completed / total_combinations
            eta = elapsed / progress - elapsed if progress > 0 else 0

            print(
                f"Progress: {completed}/{total_combinations} ({100*progress:.1f}%) - "
                f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min"
            )

            # Save intermediate results
            if save_progress and completed % (batch_size * 5) == 0:
                with open("sarima_grid_search_progress.pkl", "wb") as f:
                    pickle.dump(results, f)

        # Convert to DataFrame and analyze
        results_df = pd.DataFrame(results)

        # Filter out failed models
        valid_results = results_df[
            (results_df["converged"] == True)
            & (results_df["aic"] < np.inf)
            & (results_df["error"].isna())
        ].copy()

        if len(valid_results) == 0:
            print("⚠ No models converged successfully!")
            print("Showing all attempted results for debugging:")
            print(results_df[["order", "seasonal_order", "error"]].head(10))
            return None

        print(f"\n" + "=" * 60)
        print("GRID SEARCH RESULTS")
        print("=" * 60)
        print(f"Successfully fitted models: {len(valid_results)}/{total_combinations}")
        print(f"Total search time: {(time.time() - start_time)/60:.1f} minutes")

        # Sort by AIC
        valid_results = valid_results.sort_values("aic").reset_index(drop=True)

        # Show top 10 models
        print(f"\nTop 10 Models by AIC:")
        print("-" * 80)
        print(
            f"{'Rank':<4} {'SARIMA Order':<20} {'Seasonal Order':<20} {'AIC':<10} {'BIC':<10}"
        )
        print("-" * 80)

        for i, row in valid_results.head(10).iterrows():
            order_str = f"({row['order'][0]},{row['order'][1]},{row['order'][2]})"
            seasonal_str = f"({row['seasonal_order'][0]},{row['seasonal_order'][1]},{row['seasonal_order'][2]},{row['seasonal_order'][3]})"
            print(
                f"{i+1:<4} {order_str:<20} {seasonal_str:<20} {row['aic']:<10.2f} {row['bic']:<10.2f}"
            )

        # Store results
        self.grid_search_results = results_df
        self.best_results = valid_results
        self.best_order = valid_results.iloc[0]["order"]
        self.best_seasonal_order = valid_results.iloc[0]["seasonal_order"]

        # Save final results
        if save_progress:
            results_df.to_csv("sarima_grid_search_full_results.csv", index=False)
            valid_results.to_csv("sarima_grid_search_best_results.csv", index=False)
            print(f"\nResults saved to:")
            print(f"  - sarima_grid_search_full_results.csv")
            print(f"  - sarima_grid_search_best_results.csv")

        return valid_results

    def fit_best_model(self, train_data=None):
        """Fit the best model found by grid search"""
        if not hasattr(self, "best_order"):
            print("Run grid search first!")
            return None

        if train_data is None:
            train_data = self.temp_series[:-365]  # Hold out last year

        print(
            f"\nFitting best model: SARIMA{self.best_order} x {self.best_seasonal_order}"
        )

        try:
            model = SARIMAX(
                train_data,
                order=self.best_order,
                seasonal_order=self.best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted_model = model.fit(disp=False, maxiter=200)

            print(f"✓ Best model fitted successfully!")
            print(f"  AIC: {fitted_model.aic:.2f}")
            print(f"  BIC: {fitted_model.bic:.2f}")
            print(f"  Log-likelihood: {fitted_model.llf:.2f}")

            self.best_fitted_model = fitted_model
            return fitted_model

        except Exception as e:
            print(f"✗ Error fitting best model: {e}")
            return None

    def validate_best_model(self, test_years=2, max_horizon=30):
        """Validate the best model with walk-forward approach"""
        if not hasattr(self, "best_fitted_model"):
            print("Fit best model first!")
            return None

        print(f"\n" + "=" * 50)
        print("BEST MODEL VALIDATION")
        print("=" * 50)

        # Set up test data
        test_start_year = self.temp_series.index.max().year - test_years + 1
        test_data = self.temp_series[self.temp_series.index.year >= test_start_year]

        print(
            f"Validation period: {test_data.index.min().date()} to {test_data.index.max().date()}"
        )

        # Generate forecasts from multiple origins
        forecast_origins = test_data.index[::14]  # Every 2 weeks

        all_results = []

        for origin_date in forecast_origins:
            print(f"Forecasting from {origin_date.date()}...")

            # Expanding window training data
            current_train = self.temp_series[self.temp_series.index < origin_date]

            if len(current_train) < 730:  # Need at least 2 years
                continue

            try:
                # Refit model on current data
                model = SARIMAX(
                    current_train,
                    order=self.best_order,
                    seasonal_order=self.best_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                fitted_model = model.fit(disp=False, maxiter=100)

                # Generate forecasts
                forecasts = fitted_model.forecast(steps=max_horizon)

                # Collect results
                for horizon in range(1, max_horizon + 1):
                    forecast_date = origin_date + timedelta(days=horizon)

                    if forecast_date in test_data.index:
                        actual = test_data.loc[forecast_date]
                        predicted = forecasts[horizon - 1]

                        all_results.append(
                            {
                                "origin_date": origin_date,
                                "forecast_date": forecast_date,
                                "horizon": horizon,
                                "actual": actual,
                                "predicted": predicted,
                                "error": abs(actual - predicted),
                            }
                        )

            except Exception as e:
                print(f"  Error: {str(e)[:50]}...")
                continue

        self.validation_results = pd.DataFrame(all_results)
        print(
            f"\nValidation complete: {len(self.validation_results)} forecast-actual pairs"
        )

        return self.validation_results


def main():
    """Main execution with comprehensive grid search"""
    print("Starting comprehensive SARIMA analysis...")

    # Initialize forecaster
    forecaster = RobustSARIMAForecaster("temagami_features.csv")

    # Comprehensive grid search
    print(f"\nStarting grid search on M2 Max...")
    best_results = forecaster.comprehensive_grid_search(
        max_p=3,  # Test p = 0,1,2,3
        max_q=3,  # Test q = 0,1,2,3
        max_P=2,  # Test P = 0,1,2
        max_Q=2,  # Test Q = 0,1,2
        seasonal_period=365,
        n_jobs=8,  # Use 8 cores (M2 Max has 12, leave some headroom)
        save_progress=True,
    )

    if best_results is not None:
        # Fit the best model
        best_model = forecaster.fit_best_model()

        if best_model is not None:
            # Validate performance
            validation_results = forecaster.validate_best_model(
                test_years=2, max_horizon=30
            )

            if validation_results is not None:
                # Evaluate performance
                performance = validation_results.groupby("horizon")["error"].agg(
                    ["mean", "std", "count"]
                )
                performance.columns = ["MAE", "Std", "Count"]

                print(f"\nBest SARIMA Performance:")
                print("-" * 40)
                for horizon in [1, 3, 7, 14, 21, 30]:
                    if horizon in performance.index:
                        row = performance.loc[horizon]
                        print(
                            f"{horizon:2d} days: {row['MAE']:.2f}°C MAE ({int(row['Count'])} forecasts)"
                        )

        print(f"\nGrid search complete! Check saved CSV files for full results.")
        return forecaster, best_results

    else:
        print("Grid search failed - check the error messages above")
        return forecaster, None


if __name__ == "__main__":
    forecaster, results = main()
