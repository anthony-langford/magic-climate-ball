import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def diagnose_prophet_issue():
    """Diagnose why Prophet is giving crazy forecasts"""

    print("=" * 60)
    print("PROPHET DIAGNOSIS AND FIX")
    print("=" * 60)

    # Load data
    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)

    # Create Prophet data
    prophet_data = pd.DataFrame({"ds": df.index, "y": df["t_mean"]}).dropna()

    print(f"Data loaded: {len(prophet_data)} observations")
    print(
        f"Temperature range: {prophet_data['y'].min():.1f}°C to {prophet_data['y'].max():.1f}°C"
    )
    print(f"Temperature mean: {prophet_data['y'].mean():.1f}°C")

    # Check for data issues
    print(f"\nData quality checks:")
    print(f"- Missing values: {prophet_data['y'].isna().sum()}")
    print(f"- Infinite values: {np.isinf(prophet_data['y']).sum()}")
    print(f"- Extreme values (>50°C): {(prophet_data['y'] > 50).sum()}")
    print(f"- Extreme values (<-50°C): {(prophet_data['y'] < -50).sum()}")

    # Plot recent data to see what Prophet is working with
    recent_data = prophet_data.tail(365 * 2)  # Last 2 years

    plt.figure(figsize=(15, 6))
    plt.plot(recent_data["ds"], recent_data["y"], "b-", linewidth=1, alpha=0.7)
    plt.title("Recent Temperature Data (Input to Prophet)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Test different Prophet configurations to find what works
    configurations = [
        {
            "name": "Conservative Prophet",
            "params": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.001,  # Very conservative
                "seasonality_prior_scale": 0.1,  # Low seasonality strength
                "n_changepoints": 10,  # Few changepoints
                "growth": "linear",
                "seasonality_mode": "additive",
            },
        },
        {
            "name": "Flat Prophet",
            "params": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.001,
                "seasonality_prior_scale": 1.0,
                "growth": "flat",  # No trend allowed
            },
        },
        {
            "name": "Basic Prophet",
            "params": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
            },
        },
    ]

    # Test each configuration
    for config in configurations:
        print(f"\nTesting {config['name']}...")

        try:
            # Use recent data for testing
            test_data = prophet_data.tail(400)  # About 1 year
            train_data = test_data.iloc[:-30]  # Train on first part
            actual_test = test_data.iloc[-30:]  # Test on last 30 days

            # Fit model
            model = Prophet(**config["params"])
            model.fit(train_data)

            # Make forecast
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            # Check forecast values
            forecast_test = forecast.tail(30)
            min_forecast = forecast_test["yhat"].min()
            max_forecast = forecast_test["yhat"].max()

            print(f"  Forecast range: {min_forecast:.1f}°C to {max_forecast:.1f}°C")

            # Check if forecast is reasonable
            if min_forecast > -100 and max_forecast < 100:
                print(f"  ✓ Reasonable forecast range")

                # Calculate MAE
                mae = mean_absolute_error(actual_test["y"], forecast_test["yhat"])
                print(f"  MAE: {mae:.2f}°C")

                # Plot this working example
                plt.figure(figsize=(15, 8))

                # Training data
                plt.plot(
                    train_data["ds"],
                    train_data["y"],
                    "b-",
                    label="Training",
                    linewidth=1,
                )

                # Actual test data
                plt.plot(
                    actual_test["ds"],
                    actual_test["y"],
                    "g-",
                    label="Actual",
                    linewidth=2,
                )

                # Forecast
                plt.plot(
                    forecast_test["ds"],
                    forecast_test["yhat"],
                    "r--",
                    label="Forecast",
                    linewidth=2,
                )

                # Uncertainty
                plt.fill_between(
                    forecast_test["ds"],
                    forecast_test["yhat_lower"],
                    forecast_test["yhat_upper"],
                    alpha=0.3,
                    color="red",
                )

                plt.axvline(
                    x=train_data["ds"].iloc[-1], color="black", linestyle=":", alpha=0.7
                )
                plt.title(f'{config["name"]} - Working Example (MAE: {mae:.2f}°C)')
                plt.xlabel("Date")
                plt.ylabel("Temperature (°C)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

                return model, config  # Return the working model

            else:
                print(f"  ✗ Unreasonable forecast range!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print("\n⚠ No Prophet configuration worked properly")
    return None, None


def simple_prophet_validation():
    """Simple Prophet validation with working configuration"""

    print(f"\n" + "=" * 60)
    print("SIMPLE PROPHET VALIDATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv("temagami_features.csv", index_col=0, parse_dates=True)
    prophet_data = pd.DataFrame({"ds": df.index, "y": df["t_mean"]}).dropna()

    # Use conservative Prophet settings
    model_config = {
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.001,  # Very conservative
        "seasonality_prior_scale": 0.5,
        "growth": "flat",  # No long-term trend
    }

    # Test different forecast horizons
    horizons = [7, 14, 30]
    results = []

    # Use last 2 years for testing
    test_data = prophet_data.tail(730)

    # Test from multiple points
    test_points = test_data.index[::60]  # Every 60 days

    print(f"Testing Prophet from {len(test_points)} different starting points...")

    for i, test_point in enumerate(test_points[:-1]):  # Skip last point
        print(f"Test point {i+1}/{len(test_points)-1}: {test_point.date()}")

        # Training data up to test point
        train_data = prophet_data[prophet_data["ds"] < test_point].copy()

        if len(train_data) < 365:  # Need at least 1 year
            continue

        try:
            # Fit model
            model = Prophet(**model_config)
            model.fit(train_data)

            # Test each horizon
            for horizon in horizons:
                forecast_date = test_point + pd.Timedelta(days=horizon)

                if forecast_date in prophet_data["ds"].values:
                    # Make forecast
                    future = model.make_future_dataframe(periods=horizon)
                    forecast = model.predict(future)

                    # Get prediction and actual
                    predicted = forecast.iloc[-1]["yhat"]
                    actual_row = prophet_data[prophet_data["ds"] == forecast_date]

                    if len(actual_row) > 0:
                        actual = actual_row["y"].iloc[0]

                        # Check if prediction is reasonable
                        if -50 < predicted < 50:  # Reasonable temperature range
                            error = abs(actual - predicted)

                            results.append(
                                {
                                    "test_point": test_point,
                                    "forecast_date": forecast_date,
                                    "horizon": horizon,
                                    "actual": actual,
                                    "predicted": predicted,
                                    "error": error,
                                }
                            )

        except Exception as e:
            print(f"  Error: {str(e)[:50]}...")
            continue

    if results:
        results_df = pd.DataFrame(results)

        print(f"\nPROPHET VALIDATION RESULTS:")
        print(f"Successfully completed {len(results_df)} forecasts")
        print("-" * 50)

        # Performance by horizon
        for horizon in horizons:
            horizon_results = results_df[results_df["horizon"] == horizon]
            if len(horizon_results) > 0:
                mae = horizon_results["error"].mean()
                count = len(horizon_results)
                print(f"{horizon:2d} days: {mae:.2f}°C MAE ({count} forecasts)")

        # Compare with baselines
        try:
            baseline_df = pd.read_csv("baseline_results.csv")

            print(f"\nComparison with baselines:")
            print(
                f"{'Horizon':<8} {'Prophet':<10} {'Climatology':<12} {'Improvement':<12}"
            )
            print("-" * 45)

            for horizon in horizons:
                prophet_results = results_df[results_df["horizon"] == horizon]
                baseline_row = baseline_df[baseline_df["Horizon"] == horizon]

                if len(prophet_results) > 0 and len(baseline_row) > 0:
                    prophet_mae = prophet_results["error"].mean()
                    climatology_mae = baseline_row["Climatology_MAE"].iloc[0]
                    improvement = (
                        (climatology_mae - prophet_mae) / climatology_mae
                    ) * 100

                    print(
                        f"{horizon:<8} {prophet_mae:<10.2f} {climatology_mae:<12.2f} {improvement:+8.1f}%"
                    )

        except FileNotFoundError:
            print("Baseline results not available")

        return results_df

    else:
        print("No successful Prophet forecasts")
        return None


def main():
    """Main Prophet diagnosis and fix"""

    # First, diagnose the issue
    working_model, working_config = diagnose_prophet_issue()

    if working_model is not None:
        print(f"\n✓ Found working Prophet configuration: {working_config['name']}")

        # Run simple validation
        results = simple_prophet_validation()

        if results is not None:
            print(f"\n✓ Prophet validation successful!")
            return results
        else:
            print(f"\n⚠ Prophet validation failed")

    else:
        print(f"\n✗ Could not get Prophet to work properly")
        print("Recommendation: Use the ML approach instead")

    return None


if __name__ == "__main__":
    results = main()
