# ============================================================
# lstm_model.py - FIXED for proper direction predictions
# ============================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from data_pipeline import (
    N_FEATURES, N_OUTPUTS, SEQ_LEN,
    OPEN_IDX, CLOSE_IDX,
    inverse_transform_col
)

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_model(seq_len=SEQ_LEN, n_features=N_FEATURES, n_outputs=N_OUTPUTS):
    """
    Bidirectional LSTM with Layer Normalization.
    LayerNorm normalizes activations across features within each timestep.
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_len, n_features))),
        tf.keras.layers.Dropout(0.3),
        Bidirectional(LSTM(32)),
        tf.keras.layers.Dropout(0.3),
        LayerNormalization(),
        Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        Dense(n_outputs)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"]
    )
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, model_path=None, n_features=None):
    """
    Train with class balancing via SAMPLE WEIGHTING.
    Samples with minority direction get higher weight.
    """
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    if n_features is None:
        n_features = X_train.shape[2]
    
    model = build_model(n_features=n_features)
    
    # Calculate sample weights based on direction
    close_returns = y_train[:, 1]
    direction = (close_returns > 0).astype(int)
    
    # Sample weights: minority class gets more weight
    n_up = direction.sum()
    n_down = len(direction) - n_up
    up_weight = len(direction) / (n_up + 1)
    down_weight = len(direction) / (n_down + 1)
    
    sample_weights = np.where(direction == 1, up_weight, down_weight)
    print(f"Class weights: UP={up_weight:.2f}, DOWN={down_weight:.2f}")
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )
    
    if model_path:
        model.save(model_path)
    return model


def load_or_train(X_train, y_train, X_val=None, y_val=None, model_path=None):
    """Load saved model or train new one."""
    if model_path and os.path.exists(model_path):
        model = load_model(model_path)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="huber", metrics=["mae"])
        return model, True
    return train_lstm(X_train, y_train, X_val, y_val, model_path), False


def forecast_ohlcv(model, last_sequence, days, scaler, raw_ohlcv):
    """Forecast with autoregressive loop."""
    predictions = {"open_prices": [], "close_prices": [], "open_returns": [], "close_returns": []}
    seq = last_sequence.copy()
    
    base_prices = raw_ohlcv[-1]
    current_price = {"Open": base_prices[0], "Close": base_prices[3]}
    
    for _ in range(days):
        pred = model.predict(seq.reshape(1, -1, 8), verbose=0)[0]
        open_ret, close_ret = pred[0], pred[1]
        
        current_price["Open"] *= np.exp(open_ret)
        current_price["Close"] *= np.exp(close_ret)
        
        predictions["open_returns"].append(float(open_ret))
        predictions["close_returns"].append(float(close_ret))
        predictions["open_prices"].append(round(current_price["Open"], 2))
        predictions["close_prices"].append(round(current_price["Close"], 2))
        
        new_seq = np.roll(seq, -1, axis=0)
        new_seq[-1] = [open_ret, close_ret, current_price["Open"], current_price["Close"], 
                      base_prices[4], 0.5, 0, 0.5]
        seq = new_seq
    
    return predictions