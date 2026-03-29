# ==============================
# 📊 Financial Time Series Forecasting using CNN
# ==============================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import stft
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf


# ==============================
# 🔹 STEP 1: Get Data (Robust)
# ==============================
def get_data():
    signal = None

    # Try Yahoo Finance (AAPL)
    try:
        print("Trying Yahoo Finance (AAPL)...")
        data = yf.download("AAPL", period="5y", interval="1d")
        if not data.empty:
            print("✅ Loaded from Yahoo Finance")
            return data['Close'].values.flatten()
    except Exception as e:
        print("❌ Yahoo failed:", e)

    # Try another stock (India)
    try:
        print("Trying alternate stock (TCS.NS)...")
        data = yf.download("TCS.NS", period="5y", interval="1d")
        if not data.empty:
            print("✅ Loaded from NSE")
            return data['Close'].values.flatten()
    except Exception as e:
        print("❌ NSE failed:", e)

    # Fallback: synthetic data
    print("⚠️ Using synthetic data...")
    np.random.seed(0)
    t = np.arange(0, 1000)
    signal = np.sin(0.02 * t) + 0.5 * np.random.randn(1000)

    return signal


# ==============================
# 🔹 STEP 2: Load Signal
# ==============================
signal = get_data()
print("Signal length:", len(signal))


# ==============================
# 🔹 STEP 3: Normalize Data
# ==============================
scaler = MinMaxScaler()
signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()


# ==============================
# 🔹 STEP 4: Create Dataset (Sliding Window + STFT)
# ==============================
window_size = 128
step = 10

X = []
y = []

for i in range(0, len(signal) - window_size, step):
    segment = signal[i:i + window_size]

    # Apply STFT
    f, t, Zxx = stft(segment, nperseg=32)
    spec = np.abs(Zxx)

    # Normalize spectrogram
    if np.max(spec) != 0:
        spec = spec / np.max(spec)

    # Resize to 64x64 for CNN
    spec = cv2.resize(spec, (64, 64))

    # Add channel dimension
    spec = spec.reshape(64, 64, 1)

    X.append(spec)
    y.append(signal[i + window_size])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==============================
# 🔹 STEP 5: Build CNN Model
# ==============================
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.summary()


# ==============================
# 🔹 STEP 6: Train Model
# ==============================
model.fit(X, y, epochs=10, batch_size=16)


# ==============================
# 🔹 STEP 7: Prediction
# ==============================
pred = model.predict(X)

# Convert back to original scale
pred = scaler.inverse_transform(pred)
y_actual = scaler.inverse_transform(y.reshape(-1, 1))


# ==============================
# 🔹 STEP 8: Evaluation
# ==============================
mse = mean_squared_error(y_actual, pred)
print("MSE:", mse)


# ==============================
# 🔹 STEP 9: Plot Results
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(y_actual, label="Actual")
plt.plot(pred, label="Predicted")
plt.title("Actual vs Predicted")
plt.legend()
plt.savefig("output.png")   # Save for website
plt.show()