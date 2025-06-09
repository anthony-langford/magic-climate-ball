import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        Input,
        Conv1D,
        GlobalMaxPooling1D,
        MaxPooling1D,
    )
    from tensorflow.keras.layers import BatchNormalization, Activation, Add, Lambda
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2

    print("✓ TensorFlow/Keras available for TCN")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠ TensorFlow not available")
    TENSORFLOW_AVAILABLE = False


class TCNTemperatureForecaster:
    """Temporal Convolutional Network for temperature forecasting"""

    def __init__(self, data_path="ultra_clean_ml_features.csv"):
        """Initialize with clean temperature data"""
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        self.target_col = "t_mean"
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]

        print(f"TCN Temperature Forecaster initialized")
        print(f"Data: {len(self.df)} observations")
        print(f"Period: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"Features: {len(self.feature_cols)}")

    def create_sequences(
        self, data, sequence_length=60, forecast_horizons=[1, 7, 14, 30]
    ):
        """Create sequences for TCN training (longer sequences than LSTM)"""

        print(f"Creating TCN sequences...")
        print(f"Sequence length: {sequence_length} days (longer than LSTM)")
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
            for horizon in forecast_horizons:
                if i + horizon < len(target_scaled):
                    y_horizons.append(target_scaled[i + horizon])
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

    def residual_block(
        self, x, dilation_rate, nb_filters, kernel_size, padding="causal"
    ):
        """Create a residual block for TCN"""

        # First convolution
        conv1 = Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
        )(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = Dropout(0.2)(conv1)

        # Second convolution
        conv2 = Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
        )(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = Dropout(0.2)(conv2)

        # Residual connection
        if x.shape[-1] != nb_filters:
            # If input and output dimensions don't match, use 1x1 conv
            residual = Conv1D(nb_filters, 1, padding="same")(x)
        else:
            residual = x

        # Add residual connection
        output = Add()([conv2, residual])

        return output

    def create_tcn_models(self, input_shape, n_horizons):
        """Create different TCN architectures"""

        models = {}

        # 1. Simple TCN
        models["simple_tcn"] = {
            "name": "Simple TCN",
            "model": self._build_simple_tcn(input_shape, n_horizons),
            "description": "Basic dilated convolutions",
        }

        # 2. Deep TCN with residual connections
        models["deep_tcn"] = {
            "name": "Deep TCN",
            "model": self._build_deep_tcn(input_shape, n_horizons),
            "description": "Multiple residual blocks",
        }

        # 3. Wide TCN (more filters)
        models["wide_tcn"] = {
            "name": "Wide TCN",
            "model": self._build_wide_tcn(input_shape, n_horizons),
            "description": "More filters per layer",
        }

        print(f"Created {len(models)} TCN architectures:")
        for key, config in models.items():
            print(f"  • {config['name']}: {config['description']}")

        self.tcn_models = models
        return models

    def _build_simple_tcn(self, input_shape, n_horizons):
        """Build simple TCN model"""

        inputs = Input(shape=input_shape)

        # Dilated convolutions with increasing dilation rates
        x = Conv1D(32, 3, dilation_rate=1, padding="causal", activation="relu")(inputs)
        x = Dropout(0.2)(x)

        x = Conv1D(32, 3, dilation_rate=2, padding="causal", activation="relu")(x)
        x = Dropout(0.2)(x)

        x = Conv1D(32, 3, dilation_rate=4, padding="causal", activation="relu")(x)
        x = Dropout(0.2)(x)

        x = Conv1D(32, 3, dilation_rate=8, padding="causal", activation="relu")(x)
        x = Dropout(0.2)(x)

        # Global pooling and dense layers
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        outputs = Dense(n_horizons, activation="linear")(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def _build_deep_tcn(self, input_shape, n_horizons):
        """Build deep TCN with residual blocks"""

        inputs = Input(shape=input_shape)

        x = inputs

        # Multiple residual blocks with increasing dilation
        dilation_rates = [1, 2, 4, 8, 16]

        for dilation in dilation_rates:
            x = self.residual_block(
                x, dilation_rate=dilation, nb_filters=32, kernel_size=3
            )

        # Global pooling and output
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        outputs = Dense(n_horizons, activation="linear")(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def _build_wide_tcn(self, input_shape, n_horizons):
        """Build wide TCN with more filters"""

        inputs = Input(shape=input_shape)

        # Wider convolutions
        x = Conv1D(64, 3, dilation_rate=1, padding="causal", activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv1D(64, 3, dilation_rate=2, padding="causal", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv1D(64, 3, dilation_rate=4, padding="causal", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv1D(32, 3, dilation_rate=8, padding="causal", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Global pooling and dense layers
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        outputs = Dense(n_horizons, activation="linear")(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def train_and_validate_tcn(
        self,
        sequence_length=60,
        forecast_horizons=[1, 7, 14, 30],
        test_years=2,
        epochs=80,
        batch_size=32,
    ):
        """Train and validate TCN models"""

        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - cannot train TCN models")
            return None

        print(f"\n" + "=" * 60)
        print("TCN TRAINING AND VALIDATION")
        print("=" * 60)

        # Create sequences (longer than LSTM)
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

        # Create TCN models
        input_shape = X_train.shape[1:]
        n_horizons = len(forecast_horizons)

        self.create_tcn_models(input_shape, n_horizons)

        # Training callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        )

        # Train each model
        model_results = {}

        for model_key, model_config in self.tcn_models.items():
            print(f"\nTraining {model_config['name']}...")

            model = model_config["model"]

            # Display model architecture
            print(f"Model parameters: {model.count_params():,}")

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

        self.tcn_results = model_results
        self.forecast_horizons = forecast_horizons

        return model_results

    def analyze_tcn_performance(self):
        """Analyze TCN model performance"""

        if not hasattr(self, "tcn_results"):
            print("Train TCN models first")
            return None

        print(f"\n" + "=" * 60)
        print("TCN PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Rank models
        model_ranking = []
        for model_key, results in self.tcn_results.items():
            model_ranking.append(
                {
                    "model_key": model_key,
                    "name": results["name"],
                    "avg_mae": results["avg_mae"],
                }
            )

        model_ranking.sort(key=lambda x: x["avg_mae"])

        print(f"TCN MODEL RANKING:")
        print("-" * 40)
        print(f"{'Rank':<4} {'Model':<15} {'Avg MAE (°C)':<12}")
        print("-" * 40)

        for i, model in enumerate(model_ranking, 1):
            print(f"{i:<4} {model['name']:<15} {model['avg_mae']:<12.3f}")

        return model_ranking

    def comprehensive_comparison(self):
        """Compare TCN with all previous methods"""

        if not hasattr(self, "tcn_results"):
            print("Train TCN models first")
            return

        print(f"\n" + "=" * 60)
        print("COMPREHENSIVE METHOD COMPARISON")
        print("=" * 60)

        # Get best TCN
        best_tcn_mae = min(results["avg_mae"] for results in self.tcn_results.values())
        best_tcn_name = min(self.tcn_results.items(), key=lambda x: x[1]["avg_mae"])[1][
            "name"
        ]

        methods = [
            ("Neural Network (ML)", 2.834),
            ("Ridge Regression", 2.851),
            ("Gradient Boosting", 2.856),
            (f"{best_tcn_name} (TCN)", best_tcn_mae),
            ("Simple LSTM", 3.916),
            ("Prophet", 4.090),
            ("Climatology", 4.400),
        ]

        # Sort by performance
        methods.sort(key=lambda x: x[1])

        print(f"FINAL LEADERBOARD:")
        print("-" * 50)
        print(f"{'Rank':<4} {'Method':<25} {'MAE (°C)':<10} {'Status':<15}")
        print("-" * 50)

        for i, (method, mae) in enumerate(methods, 1):
            if "TCN" in method:
                status = "🔥 Deep Learning"
            elif "LSTM" in method:
                status = "🧠 Deep Learning"
            elif "Neural Network" in method:
                status = "🥇 Traditional ML"
            elif "Prophet" in method:
                status = "📊 Baseline"
            else:
                status = "📊 Traditional"

            print(f"{i:<4} {method:<25} {mae:<10.3f} {status:<15}")

        # Analysis
        print(f"\n🔍 TCN ANALYSIS:")
        if best_tcn_mae < 2.834:
            improvement = ((2.834 - best_tcn_mae) / 2.834) * 100
            print(f"🎉 TCN beats traditional ML by {improvement:.1f}%!")
        elif best_tcn_mae < 3.5:
            print(f"✅ TCN shows competitive deep learning performance")
        else:
            print(f"📊 TCN provides learning experience, traditional ML still wins")


def main():
    """Main TCN pipeline"""

    if not TENSORFLOW_AVAILABLE:
        print("Please install TensorFlow to run TCN models")
        return None

    print("=" * 60)
    print("TCN (TEMPORAL CONVOLUTIONAL NETWORK) PIPELINE")
    print("=" * 60)

    # Initialize TCN forecaster
    forecaster = TCNTemperatureForecaster("ultra_clean_ml_features.csv")

    # Train and validate TCN models
    tcn_results = forecaster.train_and_validate_tcn(
        sequence_length=60,  # Longer sequences than LSTM
        forecast_horizons=[1, 7, 14, 30],
        test_years=2,
        epochs=80,
        batch_size=32,
    )

    if tcn_results:
        # Analyze performance
        model_ranking = forecaster.analyze_tcn_performance()

        # Comprehensive comparison
        forecaster.comprehensive_comparison()

        print(f"\n" + "=" * 60)
        print("TCN PIPELINE COMPLETE!")
        print("=" * 60)
        print("✅ TCN models trained and validated")
        print("✅ Comparison with all previous methods")
        print("✅ Deep learning exploration complete!")

        return forecaster

    else:
        print("TCN training failed")
        return None


if __name__ == "__main__":
    forecaster = main()
