![Pepe](pepe.jpg)
![LSTM Comprehensive Analysis](lstm/lstm_comprehensive_analysis.png)

Keep going for AI-written documentation
<br><br><br><br><br><br><br>

# 🌡️ Unified Weather Forecasting System

A professional-grade, end-to-end weather forecasting pipeline that downloads historical data from Environment and Climate Change Canada (ECCC), trains neural network models, and provides temperature forecasts through both Python API and REST endpoints.

## ✨ Features

- **🚀 Complete Pipeline**: Single command goes from raw data to trained model
- **📊 Smart Caching**: Automatic data caching prevents repeated downloads
- **🧠 Neural Network**: Best-performing architecture (100-50 hidden layers)
- **📅 Multi-Horizon**: Forecasts from 1 to 30 days with confidence levels
- **🌐 REST API**: Production-ready web service with automatic documentation
- **🛶 Specialized Apps**: Built-in canoe trip planner and batch processing
- **📈 High Accuracy**: ~2.8°C Mean Absolute Error, beating traditional methods
- **🔧 Zero Configuration**: Auto-detects models and cached data

## 🏆 Performance

| Method                    | Mean Absolute Error | Notes                        |
| ------------------------- | ------------------- | ---------------------------- |
| **Neural Network (This)** | **2.78°C**          | 🥇 Best performance          |
| Prophet Baseline          | 4.09°C              | 📊 Strong traditional method |
| Climatological Baseline   | 4.40°C              | 📊 Historical averages       |
| Persistence Model         | 3.17°C              | 📊 Simple "tomorrow = today" |

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
git clone https://github.com/yourusername/weather-forecast-system
cd weather-forecast-system

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies from lock file (fastest method!)
uv sync
```

> **Why uv sync?** With a committed `uv.lock` file, `uv sync` installs exact dependency versions instantly. It ensures reproducible environments across all machines and is much faster than traditional pip workflows.

### Benefits of uv + lock file:

- ⚡ **Instant setup**: No dependency resolution needed
- 🔒 **Reproducible**: Exact same versions everywhere
- 🛡️ **Reliable**: Prevents "works on my machine" issues
- 🎯 **Precise**: Lock file contains exact hashes and versions

### Alternative Installation Methods

```bash
# Create virtual environment first (if needed)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Run Complete Pipeline

```bash
# Train model for any Canadian weather station
python weather_forecast.py --station-id 47687 --location "Temagami, ON"

# Start API server
python server.py

# REST API usage example
python usage_example.py
```

## 📋 Usage Examples

### Command Line Interface

```bash
# Complete pipeline (download → features → train → save)
python weather_forecast.py --station-id 47687 --location "Temagami, ON"

# Use existing model for forecasts
python weather_forecast.py --load-model weather_model_Temagami_ON_47687.pkl --forecast-date 2024-07-15

# Force fresh data download
python weather_forecast.py --station-id 47687 --location "Temagami, ON" --force-download

# Custom forecast horizons
python weather_forecast.py --load-model weather_model_Temagami_ON_47687.pkl --forecast-date 2024-07-15 --horizons "1,7,30"
```

### Python API

```python
from weather_forecast import WeatherForecastPipeline

# Initialize and train model
pipeline = WeatherForecastPipeline(station_id=47687, location_name="Temagami, ON")
results = pipeline.run_complete_pipeline()

# Generate forecasts
forecast = pipeline.predict_temperature("2024-07-15", horizons=[1, 7, 14, 30])

print(f"Location: {forecast['location']}")
for horizon, pred in forecast['forecasts'].items():
    print(f"{pred['horizon_days']} days: {pred['temperature']}°C ({pred['confidence']})")
```

### REST API

```bash
# Start server
python server.py --port 5001

# Single forecast
curl "http://localhost:5001/forecast?date=2024-07-15&horizons=1,7,14,30"

# Batch forecasts
curl "http://localhost:5001/batch_forecast?start_date=2024-07-01&end_date=2024-07-07"

# Model information
curl "http://localhost:5001/model_info"

# Health check
curl "http://localhost:5001/health"
```

## 🗂️ Project Structure

```
weather-forecast-system/
├── lstm                         # LSTM implementation
├── ml                           # ML implementations (neural net, gradient boosting, random forest, elastic net, etc)
├── prophet                      # Prophet implementation
├── sarima                       # SARIMA implementation
├── tcn                          # TCN implementation
├── notebooks                    # Research notebooks (data pipeline, baseline modeling, feature engineering)
├── weather_forecast.py          # Main pipeline (train models)
├── server.py                    # REST API server
├── usage_example.py             # Comprehensive examples
├── main.py                      # I say heyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
├── README.md                    # This file
├── pyproject.toml               # Modern Python dependencies
├── uv.lock                      # Locked dependency versions (committed)
├── weather_model_*.pkl          # Trained models (auto-generated)
├── weather_data_station_*.csv   # Cached weather data (auto-generated)
```

## 🌍 Finding Weather Stations

1. Visit [Environment Canada Historical Data Search](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html)
2. Search for your location
3. Click on a station with good data coverage
4. Get the Station ID from the URL (e.g., `StationID=47687`)

### Popular Canadian Stations

| Location      | Station ID | Data Range   |
| ------------- | ---------- | ------------ |
| Toronto, ON   | 48549      | 1840-present |
| Vancouver, BC | 51442      | 1870-present |
| Montreal, QC  | 50745      | 1871-present |
| Calgary, AB   | 50430      | 1881-present |
| Temagami, ON  | 47687      | 2008-present |
| Ottawa, ON    | 49568      | 1889-present |

## 🔧 API Documentation

### REST Endpoints

#### `GET /forecast`

Generate temperature forecast for a specific date.

**Parameters:**

- `date` (required): Date in YYYY-MM-DD format
- `horizons` (optional): Comma-separated forecast horizons (default: "1,3,7,14,30")

**Example:**

```bash
curl "http://localhost:5001/forecast?date=2024-07-15&horizons=1,7,14"
```

**Response:**

```json
{
  "location": "Temagami, ON",
  "station_id": 47687,
  "forecast_from": "2024-07-15",
  "generated_at": "2024-06-08T10:30:00",
  "forecasts": {
    "1_day": {
      "date": "2024-07-16",
      "temperature": 22.1,
      "horizon_days": 1,
      "confidence": "high"
    },
    "7_day": {
      "date": "2024-07-22",
      "temperature": 24.3,
      "horizon_days": 7,
      "confidence": "medium"
    }
  },
  "model_info": {
    "type": "Neural Network (MLPRegressor)",
    "expected_mae": "2.781°C"
  }
}
```

#### `GET /batch_forecast`

Generate forecasts for a date range.

**Parameters:**

- `start_date` (required): Start date in YYYY-MM-DD format
- `end_date` (required): End date in YYYY-MM-DD format
- `horizons` (optional): Comma-separated forecast horizons

#### `GET /model_info`

Get detailed information about the loaded model.

#### `GET /health`

Check server health and status.

## 🛶 Specialized Applications

### Canoe Trip Planner

```python
from weather_forecast import WeatherForecastPipeline

pipeline = WeatherForecastPipeline(47687, "Temagami, ON")
pipeline.load_model("weather_model_Temagami_ON_47687.pkl")

# Plan 5-day canoe trip
forecast = pipeline.predict_temperature("2024-07-15", horizons=[1, 2, 3, 4, 5])

for day, (horizon, pred) in enumerate(forecast['forecasts'].items(), 1):
    temp = pred['temperature']
    if temp < 15:
        gear = "🧥 Pack warm layers"
    elif temp > 25:
        gear = "🌡️ Sun protection essential"
    else:
        gear = "👕 Perfect canoeing weather"

    print(f"Day {day}: {temp}°C - {gear}")
```

### Batch Processing

```python
# Process multiple locations
stations = [
    (47687, "Temagami, ON"),
    (48549, "Toronto, ON"),
    (51442, "Vancouver, BC")
]

for station_id, location in stations:
    pipeline = WeatherForecastPipeline(station_id, location)
    results = pipeline.run_complete_pipeline()
    print(f"{location}: {results['model_performance']['test_mae']:.2f}°C MAE")
```

## 🔬 Technical Details

### Model Architecture

- **Type**: Multi-Layer Perceptron (Neural Network)
- **Hidden Layers**: 100 → 50 neurons
- **Activation**: ReLU
- **Regularization**: L2 (α=0.01)
- **Training**: Adaptive learning rate with early stopping
- **Validation**: Time-series split (last 2 years for testing)

### Feature Engineering

The system creates 13 carefully engineered features while avoiding data leakage:

**Temporal Features (4):**

- Day of year, Month
- Seasonal cycles (sin/cos transformations)

**Lag Features (4):**

- Yesterday's temperature
- Last week's temperature
- Two weeks ago temperature
- Last month's temperature

**Rolling Features (3):**

- 7-day historical average (excluding current day)
- 30-day historical average (excluding recent week)
- 14-day temperature volatility (standard deviation)

**Seasonal Features (2):**

- Winter indicator (Dec, Jan, Feb)
- Summer indicator (Jun, Jul, Aug)

### Multi-Horizon Forecasting

- **1-day**: Direct ML model prediction (highest accuracy)
- **3-7 days**: ML + seasonal trend adjustments
- **14-30 days**: ML + seasonal trends + uncertainty modeling

### Data Sources

- **Primary**: Environment and Climate Change Canada (ECCC)
- **Format**: Daily weather observations in CSV format
- **Coverage**: Canadian weather stations from 1840s to present
- **Update Frequency**: Daily (historical data is stable)

## 📊 Validation Methodology

The system employs rigorous time-series validation to ensure realistic performance estimates:

1. **Time-based Split**: Last 2 years reserved for testing (no future data leakage)
2. **Feature Generation**: Historical data only (no look-ahead bias)
3. **Cross-validation**: Multiple forecast origins tested
4. **Comparison**: Benchmarked against Prophet, climatology, and persistence

## 🐛 Troubleshooting

### Common Issues

**"No trained model found"**

```bash
# Solution: Train a model first
python weather_forecast.py --station-id 47687 --location "Temagami, ON"
```

**"No data available for date"**

- Check if date is within historical data range
- Use `--forecast-date` with dates in your dataset
- The system automatically finds the closest available date

**"Station ID not working"**

- Verify station ID at [ECCC Historical Data](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html)
- Some stations have limited data coverage
- Try a major city station for testing

**"API server not responding"**

```bash
# Check if server is running
python server.py

# Check different port
python server.py --port 5001

# Test health endpoint
curl http://localhost:5001/health
```

### Performance Issues

**Slow dependency installation**: Use uv sync for instant setup

```bash
# Fastest method - uses committed lock file
uv sync

# Alternative - still faster than pip
uv pip install pandas numpy scikit-learn requests flask
```

**Slow training**: Reduce data range

```bash
python weather_forecast.py --station-id 47687 --location "Temagami, ON" --start-year 2000
```

**Memory issues**: Use smaller time windows or consider data sampling

**Network timeouts**: Use cached data

```bash
# This will use existing cached data
python weather_forecast.py --station-id 47687 --location "Temagami, ON"
```

## 🎯 Use Cases

### ✅ Ideal Applications

- **Outdoor Activity Planning**: Camping, hiking, canoeing trips
- **Agricultural Planning**: Planting, harvesting decisions
- **Event Planning**: Outdoor weddings, festivals, sports
- **Research**: Climate analysis, trend studies
- **Education**: Weather forecasting demonstrations
- **Personal Use**: Daily temperature planning

### ⚠️ Limitations

- **Geographic**: Canadian stations only (ECCC data)
- **Variables**: Temperature only (not precipitation, wind, etc.)
- **Accuracy**: Decreases significantly beyond 14 days
- **Extreme Events**: May not predict rare weather events well
- **Microclimate**: Point forecasts, not local variations

## 🔮 Future Enhancements

### Planned Features

- [ ] **Multi-variable**: Precipitation, humidity, wind speed forecasts
- [ ] **Global Data**: Integration with other national weather services
- [ ] **Ensemble Models**: Multiple model averaging for better accuracy
- [ ] **Real-time Updates**: Automatic model retraining with new data
- [ ] **Mobile App**: React Native interface for forecasts
- [ ] **Visualization**: Interactive charts and maps
- [ ] **Alerts**: Email/SMS notifications for weather changes

### Technical Improvements

- [ ] **GPU Training**: CUDA support for faster model training
- [ ] **AutoML**: Automated hyperparameter optimization
- [ ] **Streaming**: Real-time forecast updates
- [ ] **Docker**: Containerized deployment
- [ ] **Kubernetes**: Scalable cloud deployment
- [ ] **Database**: PostgreSQL backend for large-scale data

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/weather-forecast-system
cd weather-forecast-system

# Install all dependencies (including dev dependencies)
uv sync --all-extras

# Run tests
python -m pytest tests/

# Format code
black *.py

# Run example to verify setup
python usage_example.py
```
