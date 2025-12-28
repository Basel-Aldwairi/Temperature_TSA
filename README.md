# Weather Time Series Analysis & Deep Learning Models

### Author: Basel Al-Dwairi
### Subauthor: Hazem Arar
### Subauthor: Omar Al-massi

___

## Overview
This project focuses on collecting, preprocessing, and modeling **hourly historical weather data** using both **classical Machine Learning** and **Deep Learning** approaches.

The main objectives are:
- Forecasting **hourly temperature** (regression)
- Classifying **weather conditions** (multiclass classification)
- Comparing **RNN, LSTM, and Time Series Analysis (TSA)** approaches
- Comparing 
- 

The project was developed as part of **Machine Learning and Deep Learning coursework**.

---

## Data Collection
- Weather data is collected using the **Weather Underground API**
- Hourly observations from **2010 to 2025**
- Location: **Amman, Jordan**
- Features include:
  - Temperature (°C)
  - Dew Point
  - Humidity
  - Wind Speed & Direction
  - Gust
  - Pressure
  - Weather Condition (categorical)

Raw data is stored as CSV and used as input for preprocessing pipelines.

---

## Preprocessing & Feature Engineering
Key preprocessing steps include:
- Handling missing values
- Unit conversion (Fahrenheit → Celsius)
- Cyclical encoding of wind direction (sin, cos)
- One-hot encoding of weather conditions
- Min–Max feature scaling
- Sliding window sequence generation for time series modeling

All preprocessing is modular and reproducible.

---

## Models Implemented

### 1. Temperature Forecasting (Regression)
- **Simple RNN**
- **LSTM**
- Sliding window sequences (24–72 hours)
- Metrics:
  - MAE
  - MSE
  - R² Score

### 2. Weather Condition Classification
- **LSTM-based multiclass classifier**
- One-hot encoded condition labels
- Metrics:
  - Precision
  - Recall
  - F1-score (macro)
  - Confusion Matrix

---

## Model Comparison
- RNN and LSTM models learn **temporal dependencies directly from sequences**
- Tree-based TSA models (with lag features) are discussed as strong baselines
- TSA models perform well due to autocorrelation, but may rely heavily on past values
- Sequence models offer better generalization and extensibility

---

## Repository Structure

