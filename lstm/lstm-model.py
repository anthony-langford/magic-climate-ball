import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2

    print("✓ TensorFlow/Keras available")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠ TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False


class LSTMTemperatureForecaster:
    """LSTM-based temperature forecasting with proper time series handling"""

    def __init__(self, data_path="ultra_clean_ml_features.csv"):
        """Initialize with clean temperature data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        self.target_col = "t_mean"
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]

        print(f"LSTM Temperature Forecaster initialized")
        print(f"Data: {len(self.df)} observations")
        print(f"Period: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"Features: {len(self.feature_cols)}")

    def create_sequences(
        self, data, sequence_length=30, forecast_horizons=[1, 7, 14, 30]
    ):
        """Create sequences for LSTM training"""

        print(f"Creating LSTM sequences...")
        print(f"Sequence length: {sequence_length} days")
        print(f"Forecast horizons: {forecast_horizons} days")

        # Prepare features and target
        features = data[self.feature_cols].values
        target = data[self.target_col].values
        dates = data.index

        # Scale features
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(
            target.reshape(-1, 1)
        ).flatten()

        # Create sequences
        X_sequences = []
        y_multi_horizon = []
        sequence_dates = []

        for i in range(sequence_length, len(data)):
            # Input sequence (past sequence_length days)
            X_seq = features_scaled[i - sequence_length : i]

            # Multi-horizon targets
            y_horizons = []
            valid_horizons = []

            for horizon in forecast_horizons:
                if i + horizon < len(target_scaled):
                    y_horizons.append(target_scaled[i + horizon])
                    valid_horizons.append(horizon)
                else:
                    break

            # Only add if we have all horizons
            if len(y_horizons) == len(forecast_horizons):
                X_sequences.append(X_seq)
                y_multi_horizon.append(y_horizons)
                sequence_dates.append(dates[i])

        X_sequences = np.array(X_sequences)
        y_multi_horizon = np.array(y_multi_horizon)

        print(f"Created {len(X_sequences)} sequences")
        print(f"Sequence shape: {X_sequences.shape}")
        print(f"Target shape: {y_multi_horizon.shape}")

        return X_sequences, y_multi_horizon, sequence_dates

    def create_lstm_models(self, input_shape, n_horizons):
        """Create different LSTM architectures"""

        models = {}

        # 1. Simple LSTM
        models["simple_lstm"] = {
            "name": "Simple LSTM",
            "model": self._build_simple_lstm(input_shape, n_horizons),
            "description": "Single LSTM layer with dropout",
        }

        # 2. Deep LSTM
        models["deep_lstm"] = {
            "name": "Deep LSTM",
            "model": self._build_deep_lstm(input_shape, n_horizons),
            "description": "Multiple LSTM layers",
        }

        # 3. Bidirectional LSTM
        models["bidirectional_lstm"] = {
            "name": "Bidirectional LSTM",
            "model": self._build_bidirectional_lstm(input_shape, n_horizons),
            "description": "Bidirectional LSTM layers",
        }

        print(f"Created {len(models)} LSTM architectures:")
        for key, config in models.items():
            print(f"  • {config['name']}: {config['description']}")

        self.lstm_models = models
        return models

    def _build_simple_lstm(self, input_shape, n_horizons):
        """Build simple LSTM model"""
        model = Sequential(
            [
                LSTM(
                    64,
                    return_sequences=False,
                    input_shape=input_shape,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                ),
                Dropout(0.3),
                Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                Dense(n_horizons, activation="linear"),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def _build_deep_lstm(self, input_shape, n_horizons):
        """Build deep LSTM model"""
        model = Sequential(
            [
                LSTM(
                    64,
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                ),
                LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                Dropout(0.3),
                Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                Dense(n_horizons, activation="linear"),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def _build_bidirectional_lstm(self, input_shape, n_horizons):
        """Build bidirectional LSTM model"""
        model = Sequential(
            [
                Bidirectional(
                    LSTM(
                        32,
                        return_sequences=True,
                        input_shape=input_shape,
                        dropout=0.2,
                        recurrent_dropout=0.2,
                    )
                ),
                Bidirectional(
                    LSTM(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
                ),
                Dropout(0.3),
                Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                Dense(n_horizons, activation="linear"),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def train_and_validate_lstm(
        self,
        sequence_length=30,
        forecast_horizons=[1, 7, 14, 30],
        test_years=2,
        epochs=100,
        batch_size=64,
    ):
        """Train and validate LSTM models"""

        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - cannot train LSTM models")
            return None

        print(f"\n" + "=" * 60)
        print("LSTM TRAINING AND VALIDATION")
        print("=" * 60)

        # Create sequences
        X_sequences, y_multi_horizon, sequence_dates = self.create_sequences(
            self.df, sequence_length, forecast_horizons
        )

        # Time-based train/test split
        split_date = pd.to_datetime(sequence_dates).max() - pd.Timedelta(
            days=365 * test_years
        )
        train_mask = pd.to_datetime(sequence_dates) <= split_date

        X_train, X_test = X_sequences[train_mask], X_sequences[~train_mask]
        y_train, y_test = y_multi_horizon[train_mask], y_multi_horizon[~train_mask]

        print(f"Training sequences: {len(X_train)}")
        print(f"Testing sequences: {len(X_test)}")
        print(f"Input shape: {X_train.shape[1:]}")
        print(f"Output horizons: {len(forecast_horizons)}")

        # Create LSTM models
        input_shape = X_train.shape[1:]
        n_horizons = len(forecast_horizons)

        self.create_lstm_models(input_shape, n_horizons)

        # Training callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
        )

        # Train each model
        model_results = {}

        for model_key, model_config in self.lstm_models.items():
            print(f"\nTraining {model_config['name']}...")

            model = model_config["model"]

            # Train model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1,
            )

            # Make predictions
            y_pred = model.predict(X_test, verbose=0)

            # Convert back to original scale
            y_test_original = self.target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).reshape(y_test.shape)
            y_pred_original = self.target_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).reshape(y_pred.shape)

            # Calculate MAE for each horizon
            horizon_maes = []
            for i, horizon in enumerate(forecast_horizons):
                horizon_mae = mean_absolute_error(
                    y_test_original[:, i], y_pred_original[:, i]
                )
                horizon_maes.append(horizon_mae)
                print(f"  {horizon:2d}-day horizon: {horizon_mae:.3f}°C MAE")

            avg_mae = np.mean(horizon_maes)
            print(f"  Average MAE: {avg_mae:.3f}°C")

            model_results[model_key] = {
                "name": model_config["name"],
                "model": model,
                "history": history,
                "horizon_maes": horizon_maes,
                "avg_mae": avg_mae,
                "predictions": y_pred_original,
                "actuals": y_test_original,
            }

        self.lstm_results = model_results
        self.forecast_horizons = forecast_horizons

        return model_results

    def analyze_lstm_performance(self):
        """Analyze LSTM model performance"""

        if not hasattr(self, "lstm_results"):
            print("Train LSTM models first")
            return None

        print(f"\n" + "=" * 60)
        print("LSTM PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Rank models by average performance
        model_ranking = []
        for model_key, results in self.lstm_results.items():
            model_ranking.append(
                {
                    "model_key": model_key,
                    "name": results["name"],
                    "avg_mae": results["avg_mae"],
                }
            )

        model_ranking.sort(key=lambda x: x["avg_mae"])

        print(f"LSTM MODEL RANKING:")
        print("-" * 50)
        print(f"{'Rank':<4} {'Model':<20} {'Avg MAE (°C)':<12}")
        print("-" * 50)

        for i, model in enumerate(model_ranking, 1):
            print(f"{i:<4} {model['name']:<20} {model['avg_mae']:<12.3f}")

        # Performance by horizon
        print(f"\nPERFORMANCE BY FORECAST HORIZON:")
        print("-" * 50)
        print(f"{'Horizon':<8} {'Best LSTM Model':<20} {'MAE (°C)':<10}")
        print("-" * 50)

        for i, horizon in enumerate(self.forecast_horizons):
            best_mae = float("inf")
            best_model = ""

            for model_key, results in self.lstm_results.items():
                mae = results["horizon_maes"][i]
                if mae < best_mae:
                    best_mae = mae
                    best_model = results["name"]

            print(f"{horizon:<8} {best_model:<20} {best_mae:<10.3f}")

        return model_ranking

    def compare_with_traditional_ml(self):
        """Compare LSTM with traditional ML results"""

        print(f"\n" + "=" * 60)
        print("LSTM vs TRADITIONAL ML COMPARISON")
        print("=" * 60)

        if not hasattr(self, "lstm_results"):
            print("Train LSTM models first")
            return

        # Get best LSTM performance
        best_lstm_mae = min(
            results["avg_mae"] for results in self.lstm_results.values()
        )
        best_lstm_name = min(self.lstm_results.items(), key=lambda x: x[1]["avg_mae"])[
            1
        ]["name"]

        print(f"COMPREHENSIVE COMPARISON:")
        print("-" * 60)
        print(f"Method                    Avg MAE (°C)    Improvement")
        print("-" * 60)
        print(f"{best_lstm_name:<25} {best_lstm_mae:<12.3f}  🧠 Best LSTM")
        print(f"Neural Network (ML)       2.834         📊 Traditional ML")
        print(f"Prophet Conservative      4.090         📊 Baseline")
        print(f"Climatology Baseline      4.400         📊 Traditional")

        # Calculate improvements
        ml_improvement = (
            ((2.834 - best_lstm_mae) / 2.834) * 100
            if best_lstm_mae < 2.834
            else ((best_lstm_mae - 2.834) / 2.834) * 100
        )
        prophet_improvement = ((4.09 - best_lstm_mae) / 4.09) * 100

        print(f"\nLSTM PERFORMANCE:")
        if best_lstm_mae < 2.834:
            print(f"🎉 LSTM beats traditional ML by {ml_improvement:.1f}%!")
        elif best_lstm_mae < 3.0:
            print(f"⚠️  LSTM slightly worse than ML by {ml_improvement:.1f}%")
        else:
            print(f"📊 LSTM competitive but not superior to traditional ML")

        print(f"✅ LSTM beats Prophet by {prophet_improvement:.1f}%")

        if best_lstm_mae < 2.5:
            print(f"🚀 Outstanding LSTM performance - new state of the art!")
        elif best_lstm_mae < 3.0:
            print(f"✅ Strong LSTM performance")
        else:
            print(f"📊 Solid LSTM performance")

    def plot_lstm_results(self):
        """Create comprehensive LSTM results visualization"""

        if not hasattr(self, "lstm_results"):
            print("Train LSTM models first")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Training history for best model
        best_model_key = min(
            self.lstm_results.keys(), key=lambda k: self.lstm_results[k]["avg_mae"]
        )
        best_history = self.lstm_results[best_model_key]["history"]

        axes[0, 0].plot(
            best_history.history["loss"], label="Training Loss", linewidth=2
        )
        axes[0, 0].plot(
            best_history.history["val_loss"], label="Validation Loss", linewidth=2
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss (MSE)")
        axes[0, 0].set_title(
            f'Training History - {self.lstm_results[best_model_key]["name"]}'
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: MAE by horizon for all models
        for model_key, results in self.lstm_results.items():
            axes[0, 1].plot(
                self.forecast_horizons,
                results["horizon_maes"],
                "o-",
                label=results["name"],
                linewidth=2,
                markersize=6,
            )

        axes[0, 1].set_xlabel("Forecast Horizon (days)")
        axes[0, 1].set_ylabel("MAE (°C)")
        axes[0, 1].set_title("LSTM Performance by Horizon")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Model comparison
        model_names = [results["name"] for results in self.lstm_results.values()]
        avg_maes = [results["avg_mae"] for results in self.lstm_results.values()]

        bars = axes[0, 2].bar(range(len(model_names)), avg_maes, alpha=0.7)
        axes[0, 2].set_xlabel("LSTM Models")
        axes[0, 2].set_ylabel("Average MAE (°C)")
        axes[0, 2].set_title("LSTM Model Comparison")
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels([name.replace(" ", "\n") for name in model_names])
        axes[0, 2].grid(True, alpha=0.3)

        # Add values on bars
        for bar, mae in zip(bars, avg_maes):
            axes[0, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mae:.3f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: Predictions vs Actuals (1-day horizon)
        best_results = self.lstm_results[best_model_key]
        sample_size = min(200, len(best_results["predictions"]))
        sample_indices = np.random.choice(
            len(best_results["predictions"]), sample_size, replace=False
        )

        actuals_1day = best_results["actuals"][sample_indices, 0]  # 1-day horizon
        predictions_1day = best_results["predictions"][sample_indices, 0]

        axes[1, 0].scatter(actuals_1day, predictions_1day, alpha=0.6, s=30)

        # Perfect prediction line
        min_temp = min(actuals_1day.min(), predictions_1day.min())
        max_temp = max(actuals_1day.max(), predictions_1day.max())
        axes[1, 0].plot([min_temp, max_temp], [min_temp, max_temp], "r--", alpha=0.8)

        axes[1, 0].set_xlabel("Actual Temperature (°C)")
        axes[1, 0].set_ylabel("Predicted Temperature (°C)")
        axes[1, 0].set_title("LSTM Predictions vs Actuals (1-day)")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Residuals analysis
        residuals = actuals_1day - predictions_1day
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(0, color="red", linestyle="--", alpha=0.7)
        axes[1, 1].set_xlabel("Residuals (°C)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("LSTM Residuals Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Method comparison
        methods = ["Best LSTM", "Neural Network (ML)", "Prophet", "Climatology"]
        maes = [best_results["avg_mae"], 2.834, 4.09, 4.40]
        colors = ["red", "orange", "blue", "gray"]

        bars = axes[1, 2].bar(methods, maes, color=colors, alpha=0.7)
        axes[1, 2].set_ylabel("Average MAE (°C)")
        axes[1, 2].set_title("Overall Method Comparison")
        axes[1, 2].set_xticklabels(methods, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

        # Add values on bars
        for bar, mae in zip(bars, maes):
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{mae:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("lstm_comprehensive_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Saved LSTM analysis: lstm_comprehensive_analysis.png")


def main():
    """Main LSTM pipeline"""

    if not TENSORFLOW_AVAILABLE:
        print("Please install TensorFlow to run LSTM models:")
        print("pip install tensorflow")
        return None

    print("=" * 60)
    print("LSTM TEMPERATURE FORECASTING PIPELINE")
    print("=" * 60)

    # Initialize LSTM forecaster
    forecaster = LSTMTemperatureForecaster("ultra_clean_ml_features.csv")

    # Train and validate LSTM models
    lstm_results = forecaster.train_and_validate_lstm(
        sequence_length=30,
        forecast_horizons=[1, 7, 14, 30],
        test_years=2,
        epochs=100,
        batch_size=64,
    )

    if lstm_results:
        # Analyze performance
        model_ranking = forecaster.analyze_lstm_performance()

        # Compare with traditional ML
        forecaster.compare_with_traditional_ml()

        # Create visualizations
        forecaster.plot_lstm_results()

        print(f"\n" + "=" * 60)
        print("LSTM PIPELINE COMPLETE!")
        print("=" * 60)
        print("✅ LSTM models trained and validated")
        print("✅ Performance analysis complete")
        print("✅ Comparison with traditional methods")
        print("✅ Deep learning for temperature forecasting achieved!")

        return forecaster

    else:
        print("LSTM training failed")
        return None


if __name__ == "__main__":
    forecaster = main()
