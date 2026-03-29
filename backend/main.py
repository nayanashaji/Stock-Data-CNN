from fastapi import FastAPI
import numpy as np
import cv2
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load or build model (simplified)
model = None


def get_data():
    try:
        data = yf.download("AAPL", period="2y")
        if not data.empty:
            return data['Close'].values.flatten()
    except:
        pass

    # fallback
    np.random.seed(0)
    t = np.arange(0, 1000)
    return np.sin(0.02 * t) + 0.5*np.random.randn(1000)


def process_signal(signal):
    scaler = MinMaxScaler()
    signal = scaler.fit_transform(signal.reshape(-1,1)).flatten()

    window_size = 128
    X = []

    for i in range(0, len(signal)-window_size, 10):
        segment = signal[i:i+window_size]

        f, t, Zxx = stft(segment, nperseg=32)
        spec = np.abs(Zxx)

        if np.max(spec) != 0:
            spec = spec / np.max(spec)

        spec = cv2.resize(spec, (64,64))
        spec = spec.reshape(64,64,1)

        X.append(spec)

    return np.array(X), scaler


@app.get("/predict")
def predict():
    signal = get_data()
    X, scaler = process_signal(signal)

    # Dummy prediction (since model not persisted)
    pred = np.mean(X, axis=(1,2,3))

    return {
        "prediction_sample": pred[:20].tolist()
    }