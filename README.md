# ⚡ Baseline Energy Calculator

A simple, interactive Streamlit app for analyzing electricity consumption, estimating baselines, and calculating event-based risk scores.

This tool processes uploaded Excel files, cleans the data, computes rolling-mean baselines with configurable adjustments, and generates detailed analytics including error metrics, risk scoring, and visualizations.

## 🚀 Features

📂 Data Upload & Cleaning

Upload .xlsx files

Detect available sheets

Clean and normalize column names (spaces, non-ASCII chars, casing, etc.)

Auto-detect consumption column

Convert timestamps and sort chronologically

## 📊 Baseline Calculation

Rolling mean baseline with adjustable window size

Downward adjustment factor to tune the baseline

Error metrics:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

SMAPE (Smoothed MAPE)

## ⏱️ Event-Based Analytics

Select an event time window

Compute:

% of readings above baseline

Mean & max over-baseline values

Total energy above baseline

Duration above baseline (minutes)

## 🔥 Composite Risk Score

A weighted score based on:

Frequency over baseline

Magnitude

Energy impact

Duration

Outputs:

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

📈 Visualizations

Matplotlib chart comparing actual consumption vs baseline

Interactive Streamlit data table

Difference table for event period

🛠️ Installation & Local Usage
1. Clone the repo
```bash
git clone https://github.com/NabilTallal/BaselineCalculator.git
cd BaselineCalculator
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app
```bash
streamlit run app.py
```
## 📁 Project Structure
```bash
BaselineCalculator/
│── app.py               # Main Streamlit application
│── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
## 🧪 Requirements

Python 3.8+

Streamlit

Pandas

NumPy

Matplotlib

Scikit-learn

openpyxl

(All automatically installed through requirements.txt)

## 🌐 Deployment Options

This app can be deployed easily on:

Streamlit Cloud (recommended)

Hugging Face Spaces

Render

Fly.io

Dockerized environments

If you want help deploying, just ask!

## 📌 Future Improvements (Optional)

Add sidebar layout

Add downloadable CSV report

Add anomaly detection

Add comparison with more baseline models (e.g., regression, ARIMA)

Add unit tests for baseline logic

## 👤 Author

Nabil Tallal
Computer Science Student • Data & Software Enthusiast
