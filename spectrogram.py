import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import cv2

# Generate synthetic signal
np.random.seed(0)
t_series = np.arange(0, 1000)
signal = np.sin(0.02 * t_series) + 0.5 * np.random.randn(1000)

print("Length:", len(signal))

# Sliding window dataset
window_size = 128
step = 10

X = []
y = []

for i in range(0, len(signal) - window_size, step):
    segment = signal[i:i+window_size]

    f, t, Zxx = stft(segment, nperseg=32)
    spec = np.abs(Zxx)

    # Normalize
    if np.max(spec) != 0:
        spec = spec / np.max(spec)

    # Resize to 64x64
    spec = cv2.resize(spec, (64, 64))

    # Add channel dimension
    spec = spec.reshape(64, 64, 1)

    X.append(spec)
    y.append(signal[i+window_size])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Show one sample spectrogram
plt.imshow(X[0].squeeze(), aspect='auto')
plt.title("Sample Spectrogram")
plt.colorbar()
plt.show()