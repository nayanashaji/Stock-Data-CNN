from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download("RELIANCE.NS", start="2020-01-01", end="2024-01-01")

signal = data['Close'].values

f, t, Zxx = stft(signal, nperseg=64)

plt.pcolormesh(t, f, np.abs(Zxx))
plt.title("Spectrogram")
plt.show()