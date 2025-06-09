import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def create_ultra_conservative_features():
    """Create features with absolute zero chance of data leakage"""

    print("=" * 60)
    print("ULTRA-CONSERVATIVE FEATURE CREATION")
    print("=" * 60)

    # Load raw temperature data
    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)

    # Start completely fresh - only temperature and date
    ultra_clean = pd.DataFrame(index=df.index)
    ultra_clean["t_mean"] = df["t_mean"]

    print(f"Starting with raw temperature data: {len(ultra_clean)} observations")

    # ONLY SAFE FEATURES - Nothing that could possibly leak

    # 1. TEMPORAL FEATURES ONLY (no temperature history at all)
    print("Adding temporal features...")
    ultra_clean["dayofyear"] = ultra_clean.index.dayofyear
    ultra_clean["month"] = ultra_clean.index.month
    ultra_clean["sin_doy"] = np.sin(2 * np.pi * ultra_clean["dayofyear"] / 365.25)
    ultra_clean["cos_doy"] = np.cos(2 * np.pi * ultra_clean["dayofyear"] / 365.25)
    ultra_clean["sin_month"] = np.sin(2 * np.pi * ultra_clean["month"] / 12)
    ultra_clean["cos_month"] = np.cos(2 * np.pi * ultra_clean["month"] / 12)

    # 2. ONLY VERY SAFE LAG FEATURES (minimal set)
    print("Adding minimal lag features...")
    ultra_clean["lag_1"] = ultra_clean["t_mean"].shift(1)  # Yesterday
    ultra_clean["lag_7"] = ultra_clean["t_mean"].shift(7)  # Last week
    ultra_clean["lag_365"] = ultra_clean["t_mean"].shift(365)  # Last year

    # 3. ONLY ONE ROLLING FEATURE (calculated very conservatively)
    print("Adding single rolling feature...")
    # 7-day average of temperatures from 2-8 days ago (excluding yesterday)
    ultra_clean["historical_avg_7"] = ultra_clean["t_mean"].shift(2).rolling(7).mean()

    # Clean up
    ultra_clean = ultra_clean.dropna()

    print(f"Ultra-conservative dataset: {len(ultra_clean)} observations")
    print(f"Features: {len(ultra_clean.columns)-1}")

    feature_cols = [col for col in ultra_clean.columns if col != "t_mean"]
    print("Final feature set:")
    for col in feature_cols:
        print(f"  - {col}")

    ultra_clean.to_csv("ultra_conservative_features.csv")
    print(f"\nSaved to: ultra_conservative_features.csv")

    return ultra_clean


def test_ultra_conservative_ml():
    """Test ML on ultra-conservative features"""

    print(f"\n" + "=" * 60)
    print("ULTRA-CONSERVATIVE ML TESTING")
    print("=" * 60)

    # Create ultra-conservative features
    df = create_ultra_conservative_features()

    # Prepare data
    target_col = "t_mean"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].values
    y = df[target_col].values
    dates = df.index

    print(f"Ultra-conservative ML data: {len(X)} samples, {X.shape[1]} features")

    # Very strict time-based split
    split_date = dates.max() - pd.Timedelta(days=365 * 2)  # Last 2 years for testing
    test_mask = dates > split_date

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"Training: {len(X_train)} samples (until {split_date.date()})")
    print(f"Testing: {len(X_test)} samples")

    # Simple models only
    models = {
        "Ridge (high reg)": Ridge(alpha=100.0),  # Very high regularization
        "Random Forest (simple)": RandomForestRegressor(
            n_estimators=50,
            max_depth=3,  # Very shallow
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTesting {name}...")

        if "Ridge" in name:
            # Scale for Ridge
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        # Check for impossible accuracy
        perfect_preds = np.sum(np.abs(y_test - predictions) < 0.1)
        very_good_preds = np.sum(np.abs(y_test - predictions) < 1.0)

        results[name] = {
            "mae": mae,
            "predictions": predictions,
            "perfect_rate": perfect_preds / len(y_test) * 100,
            "very_good_rate": very_good_preds / len(y_test) * 100,
        }

        print(f"  MAE: {mae:.3f}°C")
        print(
            f"  Perfect predictions (<0.1°C): {perfect_preds}/{len(y_test)} ({perfect_preds/len(y_test)*100:.1f}%)"
        )
        print(
            f"  Very good predictions (<1.0°C): {very_good_preds}/{len(y_test)} ({very_good_preds/len(y_test)*100:.1f}%)"
        )

        # Reality check
        if mae < 2.0:
            print(f"  🚨 STILL SUSPICIOUSLY GOOD!")
        elif mae < 4.0:
            print(f"  ✅ Realistic ML performance")
        else:
            print(f"  📊 Conservative performance")

    return results, y_test, dates[test_mask]


def diagnose_remaining_leakage():
    """Final diagnosis of what might still be causing leakage"""

    print(f"\n" + "=" * 60)
    print("FINAL LEAKAGE DIAGNOSIS")
    print("=" * 60)

    # Load original feature data
    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)

    print("Checking original feature engineering for subtle leakage...")

    # Check if any features are perfectly correlated with target
    target = df["t_mean"]
    correlations = {}

    for col in df.columns:
        if col != "t_mean" and df[col].dtype in [np.float64, np.float32, np.int64]:
            try:
                corr = np.corrcoef(target.dropna(), df[col].dropna())[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue

    # Sort by correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 15 correlations with target temperature:")
    print("Feature                  Correlation")
    print("-" * 40)

    for feature, corr in sorted_corr[:15]:
        print(f"{feature:<24} {corr:.6f}")

        # Flag suspicious correlations
        if corr > 0.99:
            print(f"  🚨 VERY HIGH CORRELATION - likely leakage!")
        elif corr > 0.95:
            print(f"  ⚠️  High correlation - possible leakage")

    # Check rolling mean calculations specifically
    print(f"\nChecking rolling mean calculations...")

    if "roll_mean_7" in df.columns:
        # Sample some data points
        sample_indices = [100, 500, 1000, 2000]

        for idx in sample_indices:
            if idx < len(df):
                current_temp = df["t_mean"].iloc[idx]
                stored_roll_mean = df["roll_mean_7"].iloc[idx]

                # What SHOULD the rolling mean be (excluding current value)
                correct_roll_mean = df["t_mean"].iloc[idx - 7 : idx].mean()

                print(f"Index {idx}:")
                print(f"  Current temp: {current_temp:.2f}°C")
                print(f"  Stored roll_mean_7: {stored_roll_mean:.2f}°C")
                print(f"  Correct roll_mean_7: {correct_roll_mean:.2f}°C")
                print(
                    f"  Difference: {abs(stored_roll_mean - correct_roll_mean):.4f}°C"
                )

                if abs(stored_roll_mean - current_temp) < 0.5:
                    print(f"  🚨 Rolling mean suspiciously close to current temp!")
                elif abs(stored_roll_mean - correct_roll_mean) > 0.1:
                    print(f"  ⚠️  Rolling mean calculation may be wrong")
                else:
                    print(f"  ✅ Rolling mean looks correct")
                print()


def create_baseline_only_comparison():
    """Compare against very simple baselines using same test data"""

    print(f"\n" + "=" * 60)
    print("BASELINE COMPARISON ON SAME TEST DATA")
    print("=" * 60)

    # Load the ultra-conservative data
    df = pd.read_csv("ultra_conservative_features.csv", index_col=0, parse_dates=True)

    # Same split as ML models
    split_date = df.index.max() - pd.Timedelta(days=365 * 2)
    test_mask = df.index > split_date

    test_data = df[test_mask]

    print(
        f"Test period: {test_data.index.min().date()} to {test_data.index.max().date()}"
    )
    print(f"Test samples: {len(test_data)}")

    # Simple baselines on same data
    baselines = {}

    # 1. Persistence (use yesterday's temperature)
    persistence_errors = []
    for i in range(1, len(test_data)):
        actual = test_data["t_mean"].iloc[i]
        predicted = test_data["t_mean"].iloc[i - 1]  # Yesterday
        persistence_errors.append(abs(actual - predicted))

    baselines["Persistence"] = np.mean(persistence_errors)

    # 2. Seasonal naive (use same day last year)
    seasonal_errors = []
    for i in range(len(test_data)):
        current_date = test_data.index[i]
        year_ago_date = current_date - pd.Timedelta(days=365)

        if year_ago_date in df.index:
            actual = test_data["t_mean"].iloc[i]
            predicted = df.loc[year_ago_date, "t_mean"]
            seasonal_errors.append(abs(actual - predicted))

    if seasonal_errors:
        baselines["Seasonal Naive"] = np.mean(seasonal_errors)

    # 3. Simple climatology (average for each day of year)
    train_data = df[~test_mask]
    climatology = train_data.groupby(train_data.index.dayofyear)["t_mean"].mean()

    clim_errors = []
    for i in range(len(test_data)):
        actual = test_data["t_mean"].iloc[i]
        doy = test_data.index[i].dayofyear
        if doy in climatology.index:
            predicted = climatology[doy]
            clim_errors.append(abs(actual - predicted))

    if clim_errors:
        baselines["Climatology"] = np.mean(clim_errors)

    print(f"\nBaseline performance on same test data:")
    for name, mae in baselines.items():
        print(f"{name:<15}: {mae:.3f}°C MAE")

    return baselines


def main():
    """Main ultra-conservative testing"""

    # 1. Diagnose remaining leakage sources
    diagnose_remaining_leakage()

    # 2. Test ultra-conservative ML
    ml_results, y_test, test_dates = test_ultra_conservative_ml()

    # 3. Compare with baselines on same data
    baseline_results = create_baseline_only_comparison()

    print(f"\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print("Method                    MAE (°C)")
    print("-" * 35)

    # ML results
    for name, result in ml_results.items():
        print(f"{name:<25} {result['mae']:.3f}")

    print()

    # Baseline results
    for name, mae in baseline_results.items():
        print(f"{name:<25} {mae:.3f}")

    # Final assessment
    best_ml_mae = min(result["mae"] for result in ml_results.values())
    best_baseline_mae = min(baseline_results.values())

    print(f"\nFINAL ASSESSMENT:")
    if best_ml_mae < best_baseline_mae * 0.5:
        print("🚨 ML still performing suspiciously well")
        print("   Likely remaining data leakage issues")
        print("   Recommend using Prophet + Baseline ensemble instead")
    elif best_ml_mae < best_baseline_mae:
        improvement = ((best_baseline_mae - best_ml_mae) / best_baseline_mae) * 100
        print(f"✅ ML shows realistic {improvement:.1f}% improvement over baselines")
        print("   Results appear legitimate")
    else:
        print("📊 ML performance similar to or worse than baselines")
        print("   This is actually more realistic for temperature forecasting")

    return ml_results, baseline_results


if __name__ == "__main__":
    ml_results, baseline_results = main()
