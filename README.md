# Pattern Recognition for Financial Time Series Forecasting

By Nayana Shaji Mekkunnel 

TCR24CS051    

Roll No. 50 

S4 CSE

---

## Overview

This project demonstrates how **time–frequency signal processing** and **deep learning (CNNs)** can be combined to predict values from financial time series data.

Instead of directly feeding raw time-series data into a model, the data is transformed into a **spectrogram (time–frequency representation)** and treated as an image. A **Convolutional Neural Network (CNN)** is then used to learn patterns and predict future values.

The system is further extended into a **full-stack application** with:

* A FastAPI backend for data processing and prediction
* A React frontend for visualization of results

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

* Low frequencies represent long-term trends
* High frequencies represent short-term fluctuations

---

## Technologies Used

* Python 3.10
* TensorFlow / Keras
* NumPy
* SciPy (STFT)
* Matplotlib
* OpenCV
* FastAPI
* React.js
* Chart.js

---

## Project Structure

```
Stock-Data-CNN/
│
├── backend/
│   └── main.py              # FastAPI backend (API for prediction)
│
├── frontend/
│   └── src/
│       └── App.js           # React UI (graph visualization)
│
├── model.py                 # Model training and pipeline
├── model.h5                 # Saved trained model
├── spectrogram.py           # (Optional) Spectrogram generation
├── data.py                  # (Optional) Data generation
├── requirements.txt
├── README.md
```

---

## Methodology

### 1. Data Generation / Collection

* Synthetic time-series data is generated using:

  * Trend component (sine wave)
  * Noise (random fluctuations)
* The system can also use real stock data (e.g., Yahoo Finance)

---

### 2. Time–Frequency Transformation

The **Short-Time Fourier Transform (STFT)** is applied:

* Converts signal into frequency components over time
* Produces a **spectrogram (2D representation)**

---

### 3. Data Preparation

* Sliding window applied to the signal
* Each segment converted into a spectrogram
* Resized to `64 × 64`
* Used as input to the CNN

---

### 4. CNN Model

* Convolutional layers extract spatial features
* Dense layers perform regression
* Output represents the predicted next value

---

### 5. Backend Integration

* FastAPI is used to:

  * Fetch/process data
  * Generate spectrograms
  * Run model inference
* Provides a `/predict` API endpoint

---

### 6. Frontend Visualization

* React interface allows users to:

  * Trigger prediction
  * View results as graphs
* Displays:

  * Predicted vs actual values
  * Model performance (MSE)

---

## Results

### Observations

* Model captures **overall trends** effectively
* Predictions are smoother than actual data
* High-frequency noise is harder to predict
* Performance varies depending on training data

---

## Outputs Included

* Time series plot
* Spectrogram representation
* CNN model summary
* Prediction vs actual graph (interactive)

---

## How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd Stock-Data-CNN
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run backend

```
cd backend
uvicorn main:app
```

---

### 4. Run frontend

```
cd frontend
npm install
npm start
```

---

### 5. Train model (optional)

```
python model.py
```

---

## Future Improvements

* Train model on real financial datasets
* Use advanced models (LSTM or hybrid CNN-LSTM)
* Improve prediction accuracy
* Add support for multiple stocks
* Deploy the application for public access

---

## Conclusion

This project demonstrates that:

* Financial time series can be treated as signals
* Spectrograms reveal hidden patterns
* CNNs can learn meaningful representations
* Full-stack integration enables real-time visualization of predictions

---

Completed as part of the Pattern Recognition Project
