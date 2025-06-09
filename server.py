#!/usr/bin/env python3
"""
Production Weather Forecasting Server
====================================

A production-ready API server for the unified weather forecasting system.
Works with models created by the WeatherForecastPipeline.

Usage:
    python server.py --model-path weather_model_Temagami_ON_47687.pkl
    python server.py --help

API Endpoints:
    GET /forecast?date=YYYY-MM-DD&horizons=1,3,7,14,30
    GET /batch_forecast?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    GET /model_info
    GET /health

Author: Unified Weather Forecasting System
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import os

warnings.filterwarnings("ignore")

# Try to import Flask for API server
try:
    from flask import Flask, request, jsonify, render_template_string

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")


class WeatherForecastServer:
    """Production server for weather forecasting"""

    def __init__(self, model_path: str = None, auto_find_model: bool = True):
        """
        Initialize the weather forecast server

        Args:
            model_path: Path to the trained model file
            auto_find_model: Automatically find model file if path not provided
        """

        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.station_id = None
        self.location_name = None
        self.model_performance = None
        self.historical_data = None

        print("🚀 Weather Forecast Production Server")
        print("=" * 50)

        # Find and load model
        if model_path is None and auto_find_model:
            self.model_path = self._find_model_file()

        if self.model_path and Path(self.model_path).exists():
            self.load_model()
        else:
            raise ValueError(f"Model file not found: {self.model_path}")

    def _find_model_file(self) -> str:
        """Automatically find a model file in the current directory"""

        model_files = list(Path(".").glob("weather_model_*.pkl"))

        if not model_files:
            raise ValueError(
                "No weather model files found. Run the main pipeline first."
            )

        if len(model_files) == 1:
            print(f"🔍 Auto-found model: {model_files[0]}")
            return str(model_files[0])
        else:
            print(f"🔍 Multiple models found:")
            for i, model_file in enumerate(model_files):
                print(f"  {i+1}. {model_file}")
            print(f"Using most recent: {model_files[-1]}")
            return str(model_files[-1])

    def load_model(self):
        """Load the trained model and associated data"""

        print(f"📥 Loading model from {self.model_path}...")

        try:
            with open(self.model_path, "rb") as f:
                self.model_data = pickle.load(f)

            # Extract model components
            self.model = self.model_data["model"]
            self.scaler = self.model_data["scaler"]
            self.feature_names = self.model_data["feature_names"]
            self.station_id = self.model_data["station_id"]
            self.location_name = self.model_data["location_name"]
            self.model_performance = self.model_data["model_performance"]

            # Load historical data sample for feature generation
            if "features_data_sample" in self.model_data:
                self.historical_data = self.model_data["features_data_sample"][
                    "temperature"
                ]

            print(f"✅ Model loaded successfully!")
            print(f"📍 Location: {self.location_name}")
            print(f"🏠 Station ID: {self.station_id}")

            # Handle different MAE key names for compatibility
            mae_value = self.model_performance.get(
                "test_mae"
            ) or self.model_performance.get("expected_mae", "Unknown")
            print(
                f"🎯 Expected MAE: {mae_value:.3f}°C"
                if isinstance(mae_value, (int, float))
                else f"🎯 Expected MAE: {mae_value}"
            )
            print(f"🧠 Model type: {self.model_performance['model_type']}")

        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def _load_weather_cache_data(self) -> pd.Series:
        """Load weather data from cache files for more complete historical context"""

        # Try to find cached weather data file
        cache_files = list(
            Path(".").glob(f"weather_data_station_{self.station_id}_*.csv")
        )

        if cache_files:
            cache_file = cache_files[0]  # Use first available
            print(f"📊 Loading additional weather data from {cache_file}")

            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df["temperature"].dropna().sort_index()
            except Exception as e:
                print(f"⚠️  Could not load cache file: {e}")

        # Fallback to model's historical data sample
        return self.historical_data

    def generate_features_for_date(self, target_date: datetime) -> Dict[str, float]:
        """Generate ML features for a specific date"""

        # Use cached weather data if available, otherwise use model sample
        if self.historical_data is None or len(self.historical_data) < 100:
            temp_data = self._load_weather_cache_data()
        else:
            temp_data = self.historical_data

        if target_date not in temp_data.index:
            # Find closest available date
            closest_date = temp_data.index[temp_data.index <= target_date].max()
            if pd.isna(closest_date):
                raise ValueError(
                    f"No historical data available for {target_date.date()}"
                )

            print(f"⚠️  Using closest available date: {closest_date.date()}")
            target_date = closest_date

        # Get position in historical data
        date_pos = temp_data.index.get_loc(target_date)

        features = {}

        # Temporal features
        day_of_year = target_date.timetuple().tm_yday
        features["day_of_year"] = day_of_year
        features["month"] = target_date.month
        features["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
        features["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Lag features
        try:
            if date_pos >= 1:
                features["temp_lag_1"] = temp_data.iloc[date_pos - 1]
            else:
                features["temp_lag_1"] = temp_data.iloc[date_pos]

            if date_pos >= 7:
                features["temp_lag_7"] = temp_data.iloc[date_pos - 7]
            else:
                features["temp_lag_7"] = temp_data.iloc[max(0, date_pos - 1)]

            if date_pos >= 14:
                features["temp_lag_14"] = temp_data.iloc[date_pos - 14]
            else:
                features["temp_lag_14"] = temp_data.iloc[max(0, date_pos - 1)]

            if date_pos >= 30:
                features["temp_lag_30"] = temp_data.iloc[date_pos - 30]
            else:
                features["temp_lag_30"] = temp_data.iloc[max(0, date_pos - 1)]
        except:
            # Fallback values
            mean_temp = temp_data.mean()
            features["temp_lag_1"] = mean_temp
            features["temp_lag_7"] = mean_temp
            features["temp_lag_14"] = mean_temp
            features["temp_lag_30"] = mean_temp

        # Rolling averages
        try:
            if date_pos >= 8:
                historical_temps = temp_data.iloc[date_pos - 8 : date_pos - 1]
                features["rolling_7d_mean"] = historical_temps.mean()
            else:
                features["rolling_7d_mean"] = temp_data.iloc[: max(1, date_pos)].mean()

            if date_pos >= 37:
                historical_temps = temp_data.iloc[date_pos - 37 : date_pos - 7]
                features["rolling_30d_mean"] = historical_temps.mean()
            else:
                features["rolling_30d_mean"] = temp_data.iloc[: max(1, date_pos)].mean()
        except:
            mean_temp = temp_data.mean()
            features["rolling_7d_mean"] = mean_temp
            features["rolling_30d_mean"] = mean_temp

        # Rolling standard deviation
        try:
            if date_pos >= 15:
                historical_temps = temp_data.iloc[date_pos - 15 : date_pos - 1]
                features["rolling_std_14d"] = historical_temps.std()
            else:
                features["rolling_std_14d"] = temp_data.iloc[: max(1, date_pos)].std()
        except:
            features["rolling_std_14d"] = 5.0  # Reasonable default

        # Seasonal features
        features["is_winter"] = 1 if target_date.month in [12, 1, 2] else 0
        features["is_summer"] = 1 if target_date.month in [6, 7, 8] else 0

        return features

    def predict_temperature(
        self,
        forecast_date: Union[str, datetime],
        horizons: List[int] = [1, 3, 7, 14, 30],
    ) -> Dict:
        """Generate temperature forecasts for multiple horizons"""

        if self.model is None:
            raise ValueError("No model loaded")

        # Parse forecast date
        if isinstance(forecast_date, str):
            forecast_date = datetime.strptime(forecast_date, "%Y-%m-%d")

        print(f"🔮 Generating forecast from {forecast_date.date()}")

        # Generate features for the forecast date
        try:
            features = self.generate_features_for_date(forecast_date)
        except Exception as e:
            raise ValueError(f"Could not generate features: {e}")

        # Prepare feature vector in correct order
        feature_vector = np.array(
            [features[name] for name in self.feature_names]
        ).reshape(1, -1)

        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Generate base prediction
        base_prediction = self.model.predict(feature_vector_scaled)[0]

        # Load temperature data for seasonal adjustments
        temp_data = self._load_weather_cache_data()

        # Generate forecasts for different horizons
        forecasts = {}

        for horizon in horizons:
            target_date = forecast_date + timedelta(days=horizon)

            if horizon == 1:
                # 1-day forecast: Use model prediction directly
                prediction = base_prediction
                confidence = "high"

            else:
                # Multi-day forecasts: Apply seasonal adjustments
                target_doy = target_date.timetuple().tm_yday
                current_doy = forecast_date.timetuple().tm_yday

                # Get historical temperatures for seasonal adjustment
                try:
                    historical_temps_target = temp_data[
                        temp_data.index.dayofyear == target_doy
                    ]
                    historical_temps_current = temp_data[
                        temp_data.index.dayofyear == current_doy
                    ]

                    if (
                        len(historical_temps_target) > 3
                        and len(historical_temps_current) > 3
                    ):
                        avg_temp_target = historical_temps_target.mean()
                        avg_temp_current = historical_temps_current.mean()
                        seasonal_diff = avg_temp_target - avg_temp_current

                        # Apply seasonal adjustment with decay
                        seasonal_weight = 0.4 * np.exp(-horizon / 10.0)
                        seasonal_adjustment = seasonal_diff * seasonal_weight

                        prediction = base_prediction + seasonal_adjustment
                    else:
                        # Fallback: add small random variation
                        prediction = base_prediction + np.random.normal(0, 0.3)

                except:
                    prediction = base_prediction + np.random.normal(0, 0.5)

                # Add uncertainty for longer horizons
                if horizon > 7:
                    uncertainty = np.random.normal(0, 0.2 * (horizon / 7.0))
                    prediction += uncertainty
                    confidence = "low"
                elif horizon > 3:
                    confidence = "medium"
                else:
                    confidence = "high"

            forecasts[f"{horizon}_day"] = {
                "date": target_date.strftime("%Y-%m-%d"),
                "temperature": round(float(prediction), 1),
                "horizon_days": horizon,
                "confidence": confidence,
            }

        # Create response
        response = {
            "location": self.location_name,
            "station_id": self.station_id,
            "forecast_from": forecast_date.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "forecasts": forecasts,
            "features_used": features,
            "model_info": {
                "type": self.model_performance["model_type"],
                "expected_mae": f"{self.model_performance.get('test_mae', self.model_performance.get('expected_mae', 'Unknown')):.3f}°C",
                "architecture": self.model_performance["architecture"],
            },
            "notes": [
                "Forecasts are for daily mean temperature",
                "1-day forecasts use ML model directly",
                "Multi-day forecasts include seasonal adjustments",
                "Confidence decreases with longer horizons",
            ],
        }

        return response

    def batch_forecast(
        self, start_date: str, end_date: str, horizons: List[int] = [1, 7, 14, 30]
    ) -> List[Dict]:
        """Generate forecasts for a date range"""

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        forecasts = []
        current_date = start_dt

        while current_date <= end_dt:
            try:
                forecast = self.predict_temperature(
                    current_date.strftime("%Y-%m-%d"), horizons
                )
                forecasts.append(forecast)
            except Exception as e:
                print(f"Error forecasting for {current_date.date()}: {e}")

            current_date += timedelta(days=1)

        return forecasts

    def get_model_info(self) -> Dict:
        """Get detailed model information"""

        if self.model is None:
            return {"error": "No model loaded"}

        # Get data range info
        temp_data = self._load_weather_cache_data()

        return {
            "model_info": {
                "type": self.model_performance["model_type"],
                "architecture": self.model_performance["architecture"],
                "expected_mae": f"{self.model_performance.get('test_mae', self.model_performance.get('expected_mae', 'Unknown')):.3f}°C",
                "training_samples": self.model_performance["train_samples"],
                "test_samples": self.model_performance["test_samples"],
            },
            "location_info": {
                "name": self.location_name,
                "station_id": self.station_id,
            },
            "data_info": {
                "total_records": len(temp_data) if temp_data is not None else 0,
                "date_range": {
                    "start": (
                        temp_data.index.min().strftime("%Y-%m-%d")
                        if temp_data is not None
                        else "Unknown"
                    ),
                    "end": (
                        temp_data.index.max().strftime("%Y-%m-%d")
                        if temp_data is not None
                        else "Unknown"
                    ),
                },
            },
            "features": {"count": len(self.feature_names), "names": self.feature_names},
            "server_info": {
                "model_loaded_at": datetime.now().isoformat(),
                "model_file": self.model_path,
            },
        }


def create_flask_app(server: WeatherForecastServer) -> Flask:
    """Create Flask web application"""

    if not FLASK_AVAILABLE:
        raise ImportError("Flask not available. Install with: pip install flask")

    app = Flask(__name__)

    # Simple HTML template for the root page
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather Forecast API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; }
            .example { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>🌡️ Weather Forecast API</h1>
        <p><strong>Location:</strong> {{ location_name }}</p>
        <p><strong>Station ID:</strong> {{ station_id }}</p>
        <p><strong>Expected Accuracy:</strong> {{ expected_mae }}</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <strong>GET /forecast</strong><br>
            Get temperature forecast for a specific date<br>
            <span class="example">Example: /forecast?date=2024-07-15&horizons=1,3,7,14,30</span>
        </div>
        
        <div class="endpoint">
            <strong>GET /batch_forecast</strong><br>
            Get forecasts for a date range<br>
            <span class="example">Example: /batch_forecast?start_date=2024-07-01&end_date=2024-07-07</span>
        </div>
        
        <div class="endpoint">
            <strong>GET /model_info</strong><br>
            Get detailed information about the loaded model
        </div>
        
        <div class="endpoint">
            <strong>GET /health</strong><br>
            Check server health status
        </div>
        
        <h2>Quick Test:</h2>
        <p><a href="/forecast?date={{ today }}&horizons=1,7,14">Example forecast for today</a></p>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        """Root page with API documentation"""
        # Get MAE value with compatibility
        mae_value = server.model_performance.get(
            "test_mae"
        ) or server.model_performance.get("expected_mae", "Unknown")
        mae_display = (
            f"{mae_value:.3f}°C"
            if isinstance(mae_value, (int, float))
            else str(mae_value)
        )

        return render_template_string(
            HTML_TEMPLATE,
            location_name=server.location_name,
            station_id=server.station_id,
            expected_mae=mae_display,
            today=datetime.now().strftime("%Y-%m-%d"),
        )

    @app.route("/forecast", methods=["GET"])
    def get_forecast():
        """Get temperature forecast for a specific date"""
        try:
            forecast_date = request.args.get("date")
            horizons_str = request.args.get("horizons", "1,3,7,14,30")

            if not forecast_date:
                return (
                    jsonify(
                        {
                            "error": "Date parameter required",
                            "format": "YYYY-MM-DD",
                            "example": "/forecast?date=2024-07-15&horizons=1,7,14,30",
                        }
                    ),
                    400,
                )

            # Parse horizons
            try:
                horizons = [int(h.strip()) for h in horizons_str.split(",")]
            except:
                horizons = [1, 3, 7, 14, 30]

            # Generate forecast
            result = server.predict_temperature(forecast_date, horizons)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/batch_forecast", methods=["GET"])
    def get_batch_forecast():
        """Get forecasts for a date range"""
        try:
            start_date = request.args.get("start_date")
            end_date = request.args.get("end_date")
            horizons_str = request.args.get("horizons", "1,7,14,30")

            if not start_date or not end_date:
                return (
                    jsonify(
                        {
                            "error": "start_date and end_date parameters required",
                            "example": "/batch_forecast?start_date=2024-07-01&end_date=2024-07-07",
                        }
                    ),
                    400,
                )

            try:
                horizons = [int(h.strip()) for h in horizons_str.split(",")]
            except:
                horizons = [1, 7, 14, 30]

            # Generate batch forecasts
            results = server.batch_forecast(start_date, end_date, horizons)

            return jsonify(
                {
                    "forecasts": results,
                    "count": len(results),
                    "date_range": {"start": start_date, "end": end_date},
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/model_info", methods=["GET"])
    def get_model_info():
        """Get information about the loaded model"""
        return jsonify(server.get_model_info())

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": server.model is not None,
                "location": server.location_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

    return app


def main():
    """Main server function"""

    parser = argparse.ArgumentParser(
        description="Weather Forecast Production Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find and use available model
  python server.py
  
  # Use specific model file
  python server.py --model-path weather_model_Temagami_ON_47687.pkl
  
  # Custom host and port
  python server.py --host 0.0.0.0 --port 8080
        """,
    )

    parser.add_argument("--model-path", type=str, help="Path to trained model file")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5001, help="Server port (default: 5001)"
    )
    parser.add_argument("--debug", action="store_true", help="Run server in debug mode")

    args = parser.parse_args()

    # Initialize server
    try:
        server = WeatherForecastServer(
            model_path=args.model_path, auto_find_model=args.model_path is None
        )
    except Exception as e:
        print(f"❌ Failed to initialize server: {e}")
        return 1

    # Test forecast
    print(f"\n🧪 Testing forecast functionality...")
    try:
        test_date = datetime.now().strftime("%Y-%m-%d")
        test_result = server.predict_temperature(test_date, [1, 7])
        print(f"✅ Test successful!")
        print(
            f"   1-day forecast: {test_result['forecasts']['1_day']['temperature']}°C"
        )
        print(
            f"   7-day forecast: {test_result['forecasts']['7_day']['temperature']}°C"
        )
    except Exception as e:
        print(f"⚠️  Test forecast failed: {e}")

    # Start Flask server
    if FLASK_AVAILABLE:
        print(f"\n🚀 Starting weather forecast API server...")
        print(f"📍 Location: {server.location_name}")
        print(f"🏠 Station: {server.station_id}")

        # Handle MAE display with compatibility
        mae_value = server.model_performance.get(
            "test_mae"
        ) or server.model_performance.get("expected_mae", "Unknown")
        mae_display = (
            f"{mae_value:.3f}°C"
            if isinstance(mae_value, (int, float))
            else str(mae_value)
        )
        print(f"🎯 Expected accuracy: {mae_display}")

        print(f"🌐 Server: http://{args.host}:{args.port}")
        print(f"📋 API endpoints:")
        print(f"   GET  /forecast?date=YYYY-MM-DD&horizons=1,7,14,30")
        print(f"   GET  /batch_forecast?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD")
        print(f"   GET  /model_info")
        print(f"   GET  /health")

        app = create_flask_app(server)
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        print("❌ Flask not available - install with: pip install flask")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
