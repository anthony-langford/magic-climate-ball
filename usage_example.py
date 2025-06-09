#!/usr/bin/env python3
"""
Unified Weather Forecasting System - Usage Examples
"""

import requests
import json
from datetime import datetime, timedelta
import sys
import os

# Import from the new unified system
try:
    from weather_forecast import WeatherForecastPipeline
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure you're running this from the directory containing weather_forecast.py and server.py"
    )
    sys.exit(1)


def format_temperature_display(forecast):
    """Helper function to format temperature display for both single and multi-temperature formats"""
    if "temperature_max" in forecast:
        # Multi-temperature format
        max_temp = forecast["temperature_max"]
        min_temp = forecast["temperature_min"]
        mean_temp = forecast["temperature_mean"]
        temp_range = forecast["temperature_range"]
        return f"{max_temp}°C/{min_temp}°C (avg: {mean_temp}°C, range: {temp_range}°C)"
    else:
        # Legacy single temperature format
        temp = forecast.get("temperature", "N/A")
        return f"{temp}°C"


def get_mean_temperature(forecast):
    """Helper function to get mean temperature from either format"""
    if "temperature_mean" in forecast:
        return forecast["temperature_mean"]
    elif "temperature" in forecast:
        return forecast["temperature"]
    else:
        return None


def example_1_direct_python_usage():
    """Example 1: Direct Python usage with unified pipeline"""

    print("=" * 60)
    print("EXAMPLE 1: DIRECT PYTHON USAGE")
    print("=" * 60)

    try:
        # Option A: Load existing trained model
        print("📥 Loading existing model...")

        # Find existing model file
        import glob

        model_files = glob.glob("weather_model_*.pkl")

        if not model_files:
            print("❌ No trained model found. Please run:")
            print(
                '   python weather_forecast.py --station-id 47687 --location "Temagami, ON"'
            )
            return

        model_file = model_files[0]
        print(f"Using model: {model_file}")

        # Initialize pipeline and load existing model
        pipeline = WeatherForecastPipeline(47687, "Temagami, ON")
        pipeline.load_model(model_file)

        # Single forecast
        print("\n🔮 Single Forecast:")
        result = pipeline.predict_temperature("2024-08-15")

        print(f"Location: {result['location']}")
        print(f"Forecast from: {result['forecast_from']}")
        print("Predictions:")
        for horizon, forecast in result["forecasts"].items():
            confidence = forecast["confidence"]
            confidence_emoji = (
                "🎯"
                if confidence == "high"
                else "📊" if confidence == "medium" else "🤔"
            )
            temp_display = format_temperature_display(forecast)
            print(
                f"  {horizon}: {temp_display} on {forecast['date']} {confidence_emoji}"
            )

        # Multiple horizons
        print("\n📅 Extended Forecast:")
        extended_result = pipeline.predict_temperature(
            "2024-08-15", horizons=[1, 3, 7, 14, 30]
        )

        for horizon, forecast in extended_result["forecasts"].items():
            temp_display = format_temperature_display(forecast)
            print(
                f"  {forecast['horizon_days']:2d} days: {temp_display} ({forecast['confidence']})"
            )

    except Exception as e:
        print(f"❌ Error in direct usage: {e}")
        import traceback

        traceback.print_exc()


def example_2_api_usage():
    """Example 2: API usage (requires server to be running)"""

    print("\n" + "=" * 60)
    print("EXAMPLE 2: API USAGE")
    print("=" * 60)

    base_url = "http://localhost:5001"  # Updated port

    try:
        # Health check
        print("🔍 Checking API server...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ API server is healthy")
            print(f"  Location: {health_data.get('location', 'Unknown')}")
            print(f"  Model format: {health_data.get('model_format', 'Unknown')}")
        else:
            print(f"⚠️ API health check failed: {response.status_code}")

        # Single forecast
        print("\n🔮 API Single Forecast:")
        response = requests.get(
            f"{base_url}/forecast?date=2024-08-15&horizons=1,7,14,30", timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Location: {data['location']}")
            print(f"Forecast from: {data['forecast_from']}")
            print("API Predictions:")
            for horizon, forecast in data["forecasts"].items():
                confidence = forecast["confidence"]
                confidence_emoji = (
                    "🎯"
                    if confidence == "high"
                    else "📊" if confidence == "medium" else "🤔"
                )
                temp_display = format_temperature_display(forecast)
                print(
                    f"  {horizon}: {temp_display} on {forecast['date']} {confidence_emoji}"
                )
        else:
            print(f"❌ API Error: {response.status_code}")
            if response.text:
                print(f"   Error details: {response.text}")

        # Model info
        print("\n📊 Model Information:")
        response = requests.get(f"{base_url}/model_info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            model_info = info.get("model_info", {})
            location_info = info.get("location_info", {})
            data_info = info.get("data_info", {})

            print(f"  Location: {location_info.get('name', 'Unknown')}")
            print(f"  Station ID: {location_info.get('station_id', 'Unknown')}")
            print(f"  Model: {model_info.get('type', 'Unknown')}")
            print(f"  Expected MAE: {model_info.get('expected_mae', 'Unknown')}")
            print(f"  Model Format: {model_info.get('format', 'Unknown')}")
            print(f"  Data Records: {data_info.get('total_records', 'Unknown')}")
            print(
                f"  Date Range: {data_info.get('date_range', {}).get('start', 'Unknown')} to {data_info.get('date_range', {}).get('end', 'Unknown')}"
            )

    except requests.ConnectionError:
        print("❌ API server not running. Start with:")
        print("   python server.py")
    except requests.Timeout:
        print("❌ API server timeout - it may be starting up")
    except Exception as e:
        print(f"❌ API error: {e}")


def example_3_canoe_trip_planner():
    """Example 3: Specialized canoe trip planning"""

    print("\n" + "=" * 60)
    print("EXAMPLE 3: CANOE TRIP PLANNER")
    print("=" * 60)

    try:
        # Load existing model
        import glob

        model_files = glob.glob("weather_model_*.pkl")

        if not model_files:
            print("❌ No trained model found for trip planning")
            return

        pipeline = WeatherForecastPipeline(47687, "Temagami, ON")
        pipeline.load_model(model_files[0])

        def plan_canoe_trip(trip_date: str, trip_duration: int = 5):
            """Plan a canoe trip with temperature forecasts"""

            print(
                f"\n🛶 Planning canoe trip starting {trip_date} ({trip_duration} days)"
            )
            print("-" * 50)

            # Get forecast for the trip start date
            result = pipeline.predict_temperature(trip_date, horizons=[1, 3, 7, 14, 30])

            trip_start = datetime.strptime(trip_date, "%Y-%m-%d")
            daily_forecasts = []

            # Analyze each day of the trip
            for day in range(trip_duration):
                trip_day = trip_start + timedelta(days=day)

                # Use appropriate forecast horizon
                if day == 0:
                    forecast_data = result["forecasts"]["1_day"]
                    horizon = "1-day"
                elif day <= 2:
                    forecast_data = result["forecasts"]["3_day"]
                    horizon = "3-day"
                elif day <= 6:
                    forecast_data = result["forecasts"]["7_day"]
                    horizon = "7-day"
                elif day <= 13:
                    forecast_data = result["forecasts"]["14_day"]
                    horizon = "14-day"
                else:
                    forecast_data = result["forecasts"]["30_day"]
                    horizon = "30-day"

                # Get mean temperature for analysis
                temp = get_mean_temperature(forecast_data)
                confidence = forecast_data["confidence"]

                if temp is not None:
                    daily_forecasts.append(temp)

                    # Generate recommendations
                    if temp < 5:
                        recommendation = "❄️  Very cold - bring winter gear"
                        comfort = "Challenging"
                    elif temp < 15:
                        recommendation = "🧥 Cool - pack warm layers"
                        comfort = "Good with prep"
                    elif temp < 25:
                        recommendation = "👕 Pleasant - perfect for canoeing"
                        comfort = "Excellent"
                    else:
                        recommendation = "🌡️  Hot - stay hydrated, sun protection"
                        comfort = "Good with precautions"

                    confidence_emoji = (
                        "🎯"
                        if confidence == "high"
                        else "📊" if confidence == "medium" else "🤔"
                    )

                    temp_display = format_temperature_display(forecast_data)
                    print(
                        f"Day {day+1} ({trip_day.strftime('%Y-%m-%d')}): {temp_display} ({horizon}) {confidence_emoji}"
                    )
                    print(f"  Comfort: {comfort}")
                    print(f"  Tip: {recommendation}")
                    print()

            # Overall trip assessment
            if daily_forecasts:
                avg_temp = sum(daily_forecasts) / len(daily_forecasts)
                temp_range = max(daily_forecasts) - min(daily_forecasts)

                print("📋 TRIP SUMMARY:")
                print(f"  Average temperature: {avg_temp:.1f}°C")
                print(f"  Temperature variation: {temp_range:.1f}°C")

                if temp_range > 10:
                    print(
                        "  ⚠️  High temperature variation - pack for both warm and cold"
                    )
                elif avg_temp < 10:
                    print("  🧥 Cool trip - focus on warm, waterproof gear")
                elif avg_temp > 20:
                    print("  ☀️  Warm trip - focus on sun protection and cooling")
                else:
                    print("  👌 Moderate conditions - standard canoe gear should work")

        # Example trip planning
        plan_canoe_trip("2024-07-15", 5)
        plan_canoe_trip("2024-09-01", 3)

    except Exception as e:
        print(f"❌ Error in trip planning: {e}")
        import traceback

        traceback.print_exc()


def example_4_batch_processing():
    """Example 4: Batch processing multiple dates"""

    print("\n" + "=" * 60)
    print("EXAMPLE 4: BATCH PROCESSING")
    print("=" * 60)

    try:
        # Using API for batch processing
        base_url = "http://localhost:5001"

        print("📅 Weekly batch forecast via API:")
        response = requests.get(
            f"{base_url}/batch_forecast?start_date=2024-07-01&end_date=2024-07-07&horizons=1,7",
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Generated {data['count']} forecasts:")

            for forecast in data["forecasts"]:
                date = forecast["forecast_from"]
                forecast_1 = forecast["forecasts"]["1_day"]
                forecast_7 = forecast["forecasts"]["7_day"]

                temp_1_display = format_temperature_display(forecast_1)
                temp_7_display = format_temperature_display(forecast_7)

                conf_1 = forecast_1["confidence"]
                conf_7 = forecast_7["confidence"]

                print(
                    f"  {date}: 1-day={temp_1_display} ({conf_1}), 7-day={temp_7_display} ({conf_7})"
                )

        else:
            print(f"❌ Batch API Error: {response.status_code}")
            # Fallback to direct Python approach
            print("\n🔄 Falling back to direct Python batch processing...")
            example_4_direct_batch()

    except requests.ConnectionError:
        print("❌ API server not running, using direct Python approach...")
        example_4_direct_batch()
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        import traceback

        traceback.print_exc()


def example_4_direct_batch():
    """Direct Python batch processing fallback"""

    try:
        import glob

        model_files = glob.glob("weather_model_*.pkl")

        if not model_files:
            print("❌ No trained model found for batch processing")
            return

        pipeline = WeatherForecastPipeline(47687, "Temagami, ON")
        pipeline.load_model(model_files[0])

        print("📅 Direct Python batch processing:")

        # Generate forecasts for a week
        start_date = datetime(2024, 7, 1)
        for i in range(7):
            forecast_date = start_date + timedelta(days=i)
            date_str = forecast_date.strftime("%Y-%m-%d")

            try:
                result = pipeline.predict_temperature(date_str, horizons=[1, 7])
                forecast_1 = result["forecasts"]["1_day"]
                forecast_7 = result["forecasts"]["7_day"]

                temp_1_display = format_temperature_display(forecast_1)
                temp_7_display = format_temperature_display(forecast_7)

                conf_1 = forecast_1["confidence"]
                conf_7 = forecast_7["confidence"]

                print(
                    f"  {date_str}: 1-day={temp_1_display} ({conf_1}), 7-day={temp_7_display} ({conf_7})"
                )
            except Exception as e:
                print(f"  {date_str}: Error - {e}")

    except Exception as e:
        print(f"❌ Direct batch processing error: {e}")
        import traceback

        traceback.print_exc()


def example_5_server_integration():
    """Example 5: Complete server integration test"""

    print("\n" + "=" * 60)
    print("EXAMPLE 5: SERVER INTEGRATION TEST")
    print("=" * 60)

    base_url = "http://localhost:5001"

    # Test all API endpoints
    endpoints = [
        ("/health", "Health Check"),
        ("/model_info", "Model Information"),
        ("/forecast?date=2024-08-15&horizons=1,3,7", "Single Forecast"),
    ]

    for endpoint, description in endpoints:
        try:
            print(f"\n🔍 Testing {description}:")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)

            if response.status_code == 200:
                print(f"  ✅ Success ({response.status_code})")

                # Show sample of response data
                try:
                    data = response.json()
                    if endpoint.startswith("/forecast"):
                        forecasts = data.get("forecasts", {})
                        print(f"  📊 Forecasts generated: {len(forecasts)}")

                        # Show a sample forecast with proper formatting
                        if forecasts:
                            sample_key = list(forecasts.keys())[0]
                            sample_forecast = forecasts[sample_key]
                            temp_display = format_temperature_display(sample_forecast)
                            print(f"  📋 Sample ({sample_key}): {temp_display}")

                    elif endpoint.startswith("/model_info"):
                        model_info = data.get("model_info", {})
                        print(f"  🧠 Model type: {model_info.get('type', 'Unknown')}")
                        print(
                            f"  🎯 Model format: {model_info.get('format', 'Unknown')}"
                        )
                    elif endpoint.startswith("/health"):
                        print(f"  💚 Status: {data.get('status', 'Unknown')}")
                        print(
                            f"  🔧 Model format: {data.get('model_format', 'Unknown')}"
                        )
                except Exception as inner_e:
                    print(f"  📄 Response received (parsing error: {inner_e})")
            else:
                print(f"  ❌ Failed ({response.status_code})")

        except requests.ConnectionError:
            print(f"  ❌ Connection failed - server not running")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")


def main():
    """Run all examples"""

    print("🌡️  UNIFIED WEATHER FORECASTING SYSTEM - EXAMPLES")
    print("=" * 60)
    print("This demonstrates the complete unified weather forecasting system.")
    print()

    # Check if we have required files
    import glob

    model_files = glob.glob("weather_model_*.pkl")
    cache_files = glob.glob("weather_data_station_*.csv")

    print("📋 System Status:")
    print(f"  Models found: {len(model_files)}")
    print(f"  Cache files: {len(cache_files)}")

    if not model_files:
        print("\n❌ No trained models found!")
        print("Please run the pipeline first:")
        print(
            '  python weather_forecast.py --station-id 47687 --location "Temagami, ON"'
        )
        return

    # Run examples
    example_1_direct_python_usage()
    example_2_api_usage()
    example_3_canoe_trip_planner()
    example_4_batch_processing()
    example_5_server_integration()

    print("\n" + "=" * 60)
    print("🎯 UNIFIED SYSTEM FEATURES DEMONSTRATED")
    print("=" * 60)
    print("Your unified weather forecasting system includes:")
    print("✅ Single-command pipeline (download → features → train → predict)")
    print("✅ Automatic data caching (no repeated downloads)")
    print("✅ Multi-horizon forecasting (1 to 30 days)")
    print("✅ Multi-temperature predictions (max/min/mean)")
    print("✅ Confidence indicators (high/medium/low)")
    print("✅ Direct Python API")
    print("✅ RESTful web service")
    print("✅ Batch processing capabilities")
    print("✅ Specialized applications (canoe trip planning)")
    print("✅ Model persistence and loading")
    print("✅ Seasonal trend adjustments")

    print(f"\n🚀 Complete Usage Workflow:")
    print(
        '1. Train model: python weather_forecast.py --station-id 47687 --location "Temagami, ON"'
    )
    print("2. Start server: python server.py")
    print("3. Use examples: python usage_example.py")
    print('4. Make API calls: curl "http://localhost:5001/forecast?date=2024-07-15"')
    print("5. Build applications using the Python or API interface")


if __name__ == "__main__":
    main()
