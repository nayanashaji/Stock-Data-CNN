import numpy as np
import cv2
from scipy.signal import stft
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import numpy as np
import yfinance as yf

def get_data():
    signal = None

    # ===== TRY 1: Yahoo Finance =====
    try:
        print("Trying Yahoo Finance...")
        data = yf.download("AAPL", period="5y", interval="1d")
        if not data.empty:
            signal = data['Close'].values.flatten()
            print("✅ Loaded from Yahoo Finance")
            return signal
    except Exception as e:
        print("❌ Yahoo failed:", e)

    # ===== TRY 2: Another ticker =====
    try:
        print("Trying alternate stock...")
        data = yf.download("TCS.NS", period="5y", interval="1d")
        if not data.empty:
            signal = data['Close'].values.flatten()
            print("✅ Loaded from NSE stock")
            return signal
    except Exception as e:
        print("❌ NSE failed:", e)

    # ===== FALLBACK: Synthetic Data =====
    print("⚠️ Using synthetic data...")
    np.random.seed(0)
    t = np.arange(0, 1000)
    signal = np.sin(0.02 * t) + 0.5 * np.random.randn(1000)

    return signal

# ====== STEP 1: Generate synthetic signal ======
signal = get_data()
print("Signal length:", len(signal))

# ====== STEP 2: Create dataset ======
window_size = 128
step = 10

X = []
y = []

for i in range(0, len(signal) - window_size, step):
    segment = signal[i:i+window_size]

    f, t, Zxx = stft(segment, nperseg=32)
    spec = np.abs(Zxx)

    if np.max(spec) != 0:
        spec = spec / np.max(spec)

    spec = cv2.resize(spec, (64, 64))
    spec = spec.reshape(64, 64, 1)

    X.append(spec)
    y.append(signal[i+window_size])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ====== STEP 3: Build model ======
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ====== STEP 4: Train ======
model.fit(X, y, epochs=5, batch_size=16)

# ====== STEP 5: Predict ======
pred = model.predict(X)

# ====== STEP 6: Plot ======
plt.plot(y, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.show()