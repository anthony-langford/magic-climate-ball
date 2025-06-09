import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def step_by_step_debug():
    """Step-by-step debugging to find the exact leakage source"""

    print("=" * 60)
    print("SYSTEMATIC ML DEBUGGING - LEARNING EXERCISE")
    print("=" * 60)

    # Load original data
    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)

    print(f"Original data: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Step 1: Test with ONLY the target variable (impossible case)
    print(f"\n" + "=" * 50)
    print("STEP 1: Test with only temporal features (no temperature history)")
    print("=" * 50)

    # Create dataset with NO temperature history at all
    temporal_only = pd.DataFrame(index=df.index)
    temporal_only["t_mean"] = df["t_mean"]
    temporal_only["dayofyear"] = temporal_only.index.dayofyear
    temporal_only["month"] = temporal_only.index.month
    temporal_only["sin_doy"] = np.sin(2 * np.pi * temporal_only["dayofyear"] / 365.25)
    temporal_only["cos_doy"] = np.cos(2 * np.pi * temporal_only["dayofyear"] / 365.25)

    # Test this minimal dataset
    result_1 = test_single_dataset(temporal_only, "Temporal Only")

    # Step 2: Add ONLY lag_1 (yesterday's temperature)
    print(f"\n" + "=" * 50)
    print("STEP 2: Add lag_1 (yesterday's temperature)")
    print("=" * 50)

    with_lag1 = temporal_only.copy()
    with_lag1["lag_1"] = with_lag1["t_mean"].shift(1)
    with_lag1 = with_lag1.dropna()

    result_2 = test_single_dataset(with_lag1, "Temporal + Lag1")

    # Step 3: Add lag_7 (last week)
    print(f"\n" + "=" * 50)
    print("STEP 3: Add lag_7 (last week's temperature)")
    print("=" * 50)

    with_lag7 = with_lag1.copy()
    with_lag7["lag_7"] = df["t_mean"].shift(7).reindex(with_lag7.index)
    with_lag7 = with_lag7.dropna()

    result_3 = test_single_dataset(with_lag7, "Temporal + Lag1 + Lag7")

    # Step 4: Add a MANUALLY calculated rolling mean (very careful)
    print(f"\n" + "=" * 50)
    print("STEP 4: Add manually calculated rolling mean")
    print("=" * 50)

    with_manual_roll = with_lag7.copy()

    # Calculate rolling mean VERY carefully - excluding current day
    manual_roll_mean = []
    temp_series = df["t_mean"]

    for i, date in enumerate(with_manual_roll.index):
        # Find position in original series
        orig_pos = df.index.get_loc(date)

        # Get 7 days BEFORE this date (excluding current day)
        if orig_pos >= 7:
            historical_temps = temp_series.iloc[
                orig_pos - 7 : orig_pos
            ]  # 7 days before, not including current
            manual_roll_mean.append(historical_temps.mean())
        else:
            manual_roll_mean.append(np.nan)

    with_manual_roll["manual_roll_7"] = manual_roll_mean
    with_manual_roll = with_manual_roll.dropna()

    result_4 = test_single_dataset(with_manual_roll, "With Manual Rolling Mean")

    # Step 5: Compare with Prophet-style approach
    print(f"\n" + "=" * 50)
    print("STEP 5: Prophet-style validation (single train/test split)")
    print("=" * 50)

    result_5 = test_prophet_style(with_manual_roll)

    # Summary
    print(f"\n" + "=" * 60)
    print("DEBUGGING SUMMARY")
    print("=" * 60)

    results = [
        ("Temporal Only", result_1),
        ("+ Lag1", result_2),
        ("+ Lag7", result_3),
        ("+ Manual Rolling", result_4),
        ("Prophet-style", result_5),
    ]

    print("Step                    MAE (°C)    Status")
    print("-" * 50)

    for name, mae in results:
        if mae < 1.0:
            status = "🚨 SUSPICIOUS"
        elif mae < 3.0:
            status = "⚠️  Still good"
        else:
            status = "✅ Realistic"

        print(f"{name:<20} {mae:<10.2f} {status}")

    return results


def test_single_dataset(df, name):
    """Test ML on a single dataset with proper time series validation"""

    print(f"Testing: {name}")
    print(f"Features: {[col for col in df.columns if col != 't_mean']}")
    print(f"Samples: {len(df)}")

    # Prepare features
    feature_cols = [col for col in df.columns if col != "t_mean"]
    X = df[feature_cols].values
    y = df["t_mean"].values
    dates = df.index

    # Time-based split (last year for testing)
    split_date = dates.max() - pd.Timedelta(days=365)
    test_mask = dates > split_date

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Simple Ridge regression
    model = Ridge(alpha=1.0)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)

    # Check for perfect predictions
    perfect_count = np.sum(np.abs(y_test - predictions) < 0.1)
    perfect_rate = perfect_count / len(y_test) * 100

    print(f"MAE: {mae:.3f}°C")
    print(
        f"Perfect predictions (<0.1°C): {perfect_count}/{len(y_test)} ({perfect_rate:.1f}%)"
    )

    # Show some actual predictions vs reality
    print("Sample predictions:")
    for i in range(min(5, len(y_test))):
        print(
            f"  Actual: {y_test[i]:.2f}°C, Predicted: {predictions[i]:.2f}°C, Error: {abs(y_test[i] - predictions[i]):.3f}°C"
        )

    return mae


def test_prophet_style(df):
    """Test using Prophet's validation style (single train/test, no walk-forward)"""

    print("Prophet-style validation:")
    print("- Single train/test split")
    print("- Test various horizons on same test set")
    print("- No walk-forward complexity")

    # Use same split as Prophet
    split_date = df.index.max() - pd.Timedelta(days=365 * 2)  # 2 years test
    train_data = df[df.index <= split_date].copy()
    test_data = df[df.index > split_date].copy()

    print(f"Train: {len(train_data)} samples (until {split_date.date()})")
    print(f"Test: {len(test_data)} samples")

    # Prepare training data
    feature_cols = [col for col in df.columns if col != "t_mean"]
    X_train = train_data[feature_cols].values
    y_train = train_data["t_mean"].values

    # Fit model once
    model = Ridge(alpha=10.0)  # Higher regularization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)

    # Test different horizons
    horizons = [1, 7, 14, 30]
    horizon_results = {}

    for horizon in horizons:
        horizon_errors = []

        # Test every 14 days to get good sample
        test_origins = test_data.index[::14]

        for origin_date in test_origins[:-5]:  # Leave some buffer
            forecast_date = origin_date + pd.Timedelta(days=horizon)

            if forecast_date in test_data.index:
                # Get features for forecast date
                if forecast_date in df.index:
                    feature_row = df.loc[forecast_date, feature_cols].values.reshape(
                        1, -1
                    )

                    # Make prediction
                    feature_row_scaled = scaler.transform(feature_row)
                    prediction = model.predict(feature_row_scaled)[0]
                    actual = test_data.loc[forecast_date, "t_mean"]

                    error = abs(actual - prediction)
                    horizon_errors.append(error)

        if horizon_errors:
            horizon_mae = np.mean(horizon_errors)
            horizon_results[horizon] = horizon_mae
            print(
                f"  {horizon:2d}-day horizon: {horizon_mae:.3f}°C MAE ({len(horizon_errors)} forecasts)"
            )

    # Return average across horizons
    if horizon_results:
        avg_mae = np.mean(list(horizon_results.values()))
        print(f"Average across horizons: {avg_mae:.3f}°C")
        return avg_mae
    else:
        return 999.0


def analyze_feature_leakage():
    """Analyze specific features for potential leakage"""

    print(f"\n" + "=" * 60)
    print("FEATURE LEAKAGE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)

    # Check rolling mean calculation in detail
    print("Checking rolling mean calculations...")

    # Sample a few dates and manually verify rolling mean calculation
    sample_dates = df.index[400:405]  # Pick 5 consecutive dates

    for date in sample_dates:
        current_temp = df.loc[date, "t_mean"]

        if "roll_mean_7" in df.columns:
            stored_roll_mean = df.loc[date, "roll_mean_7"]

            # Calculate what it SHOULD be
            date_pos = df.index.get_loc(date)

            # Method 1: Rolling mean including current day (WRONG)
            wrong_calc = (
                df["t_mean"].iloc[date_pos - 6 : date_pos + 1].mean()
            )  # Includes current day

            # Method 2: Rolling mean excluding current day (RIGHT)
            if date_pos >= 7:
                right_calc = (
                    df["t_mean"].iloc[date_pos - 7 : date_pos].mean()
                )  # Excludes current day
            else:
                right_calc = np.nan

            print(f"\nDate: {date.date()}")
            print(f"  Current temp: {current_temp:.2f}°C")
            print(f"  Stored roll_mean_7: {stored_roll_mean:.2f}°C")
            print(f"  Wrong calc (includes current): {wrong_calc:.2f}°C")
            print(f"  Right calc (excludes current): {right_calc:.2f}°C")
            print(f"  Diff from current: {abs(stored_roll_mean - current_temp):.3f}°C")

            if abs(stored_roll_mean - wrong_calc) < 0.001:
                print(f"  🚨 LEAKAGE: Rolling mean includes current day!")
            elif abs(stored_roll_mean - right_calc) < 0.001:
                print(f"  ✅ Correct: Rolling mean excludes current day")
            else:
                print(f"  ❓ Unclear how rolling mean was calculated")


def create_truly_clean_ml():
    """Create ML model with absolute certainty of no leakage"""

    print(f"\n" + "=" * 60)
    print("CREATING TRULY CLEAN ML MODEL")
    print("=" * 60)

    # Start completely fresh
    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)

    # Create minimal feature set from scratch
    clean_df = pd.DataFrame(index=df.index)
    clean_df["t_mean"] = df["t_mean"]

    # Only add features we create ourselves with 100% certainty
    print("Creating features from scratch...")

    # 1. Temporal features (safe)
    clean_df["day_of_year"] = clean_df.index.dayofyear
    clean_df["month"] = clean_df.index.month
    clean_df["sin_doy"] = np.sin(2 * np.pi * clean_df["day_of_year"] / 365.25)
    clean_df["cos_doy"] = np.cos(2 * np.pi * clean_df["day_of_year"] / 365.25)

    # 2. Lag features (manually created)
    clean_df["yesterday"] = clean_df["t_mean"].shift(1)
    clean_df["last_week"] = clean_df["t_mean"].shift(7)

    # 3. VERY careful rolling average (from scratch)
    rolling_avg = []
    for i in range(len(clean_df)):
        if i >= 7:  # Need at least 7 previous days
            # Average of days 2-8 ago (not including yesterday or today)
            historical_temps = clean_df["t_mean"].iloc[
                i - 8 : i - 1
            ]  # 7 days, excluding yesterday and today
            rolling_avg.append(historical_temps.mean())
        else:
            rolling_avg.append(np.nan)

    clean_df["historical_avg"] = rolling_avg

    # Remove rows with NaN
    clean_df = clean_df.dropna()

    print(f"Clean dataset: {len(clean_df)} samples")
    print(f"Features: {[col for col in clean_df.columns if col != 't_mean']}")

    # Test this ultra-clean dataset
    mae = test_single_dataset(clean_df, "Ultra-Clean from Scratch")

    # Save for inspection
    clean_df.to_csv("ultra_clean_ml_features.csv")
    print(f"Saved to: ultra_clean_ml_features.csv")

    return clean_df, mae


def main():
    """Main debugging pipeline"""

    print("Starting systematic ML debugging...")
    print("This will teach us exactly where leakage occurs and how to fix it.")

    # Step 1: Systematic debugging
    step_results = step_by_step_debug()

    # Step 2: Analyze specific feature leakage
    analyze_feature_leakage()

    # Step 3: Create truly clean model
    clean_df, clean_mae = create_truly_clean_ml()

    print(f"\n" + "=" * 60)
    print("LEARNING SUMMARY")
    print("=" * 60)

    print("Key lessons learned:")
    print("1. Time series validation is much trickier than regular ML")
    print("2. Even 'obvious' features like rolling means can leak")
    print("3. Walk-forward validation adds complexity that can introduce bugs")
    print("4. Simple train/test splits (like Prophet uses) are often more reliable")

    if clean_mae > 2.0:
        print(f"\n✅ SUCCESS: Clean ML achieved realistic {clean_mae:.2f}°C MAE")
        print("   Now we can build proper ML models with confidence!")
    else:
        print(f"\n🚨 Still need more debugging - MAE too good: {clean_mae:.2f}°C")
        print("   The validation methodology itself may have issues")

    print(f"\nNext steps:")
    print("- If clean MAE > 2°C: Build proper ML ensemble")
    print("- If clean MAE < 1°C: Focus on validation methodology")
    print("- Either way: We've learned crucial time series ML lessons!")

    return step_results, clean_df


if __name__ == "__main__":
    results, clean_data = main()
