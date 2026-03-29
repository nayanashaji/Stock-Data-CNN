# 📊 Pattern Recognition for Financial Time Series Forecasting

 

By Nayana Shaji Mekkunnel

TCR24CS051

Roll No. 50

S4 CSE

  

## Overview

This project demonstrates how **time–frequency signal processing** and **deep learning (CNNs)** can be combined to predict values from financial time series data.

Instead of directly feeding raw time-series data into a model, we transform it into a **spectrogram (time–frequency representation)** and treat it like an image. A **Convolutional Neural Network (CNN)** is then used to learn patterns and predict future values.

---

## Key Concept

Traditional time-series:

```
Price vs Time
```

Transformed approach:

```
Signal → STFT → Spectrogram → CNN → Prediction
```

* Low frequencies → long-term trends
* High frequencies → short-term fluctuations

---

## Technologies Used

* Python 3.10
* TensorFlow / Keras
* NumPy
* SciPy (STFT)
* Matplotlib
* OpenCV

---

## Project Structure

```
Stock-Data-CNN/
│
├── model.py              # Full pipeline (data → spectrogram → CNN → prediction)
├── spectrogram.py        # (Optional) Spectrogram generation
├── data.py               # (Optional) Data generation
├── requirements.txt      # Dependencies
├── .gitignore            # Ignored files
```

---

## Methodology

### 1. Signal Generation

Synthetic time-series data is generated to simulate financial behavior:

* Trend component (sine wave)
* Noise (random fluctuations)

---

### 2. Time–Frequency Transformation

We apply **Short-Time Fourier Transform (STFT)**:

* Converts signal into frequency components over time
* Produces a **spectrogram (2D representation)**

---

### 3. Data Preparation

* Sliding window applied to signal
* Each segment → spectrogram
* Resized to `64 × 64`
* Used as CNN input

---

### 4. CNN Model

* Convolutional layers extract spatial patterns
* Dense layers perform regression
* Output = predicted next value

---

### 5. Evaluation

* Predictions compared with actual values
* Metric used: **Mean Squared Error (MSE)**

---

## Results

### Observations:

* Model captures **overall trends** effectively
* Predictions are smoother than actual data
* High-frequency noise is harder to predict

---

## Outputs Included

* Time series plot
* Frequency spectrum
* Spectrogram
* CNN architecture summary
* Prediction vs actual graph

---

## How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd Stock-Data-CNN
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the project

```
python model.py
```

---

## Future Improvements

* Use real stock data (Yahoo Finance / NSE)
* Add multiple features (multivariate signals)
* Experiment with different window sizes
* Improve model architecture

---

## Conclusion

This project shows that:

* Financial time series can be treated as signals
* Spectrograms reveal hidden patterns
* CNNs can learn meaningful representations for prediction

---

Completed as part of Pattern Recognition Project
