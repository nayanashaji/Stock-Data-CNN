from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
from tensorflow.keras.models import load_model

model = load_model("../model.h5")

app = FastAPI()

# ✅ Enable CORS (important for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# 🔹 Get Data (Robust)
# ==============================
def get_data():
    try:
        data = yf.download("AAPL", period="2y")
        if not data.empty:
            print("✅ Loaded real data")
            return data['Close'].values.flatten()
    except:
        pass

    # fallback
    print("⚠️ Using synthetic data")
    np.random.seed(0)
    t = np.arange(0, 1000)
    return np.sin(0.02 * t) + 0.5 * np.random.randn(1000)


# ==============================
# 🔹 Process Signal → Spectrogram
# ==============================
def process_signal(signal):
    scaler = MinMaxScaler()
    norm_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    window_size = 128
    X = []
    y = []

    for i in range(0, len(norm_signal) - window_size, 10):
        segment = norm_signal[i:i + window_size]

        f, t, Zxx = stft(segment, nperseg=32)
        spec = np.abs(Zxx)

        if np.max(spec) != 0:
            spec = spec / np.max(spec)

        spec = cv2.resize(spec, (64, 64))
        spec = spec.reshape(64, 64, 1)

        X.append(spec)
        y.append(norm_signal[i + window_size])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


# ==============================
# 🔹 API Endpoint
# ==============================
@app.get("/predict")
def predict():
    signal = get_data()
    X, y, scaler = process_signal(signal)

    
    pred = model.predict(X).flatten()

    # Convert back to original scale
    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # Compute MSE
    mse = mean_squared_error(actual[:len(pred)], pred)

    return {
        "predicted": pred[:50].tolist(),
        "actual": actual[:50].tolist(),
        "mse": float(mse)
    }