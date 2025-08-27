# 📈 ZIDIO Stock Forecasting Dashboard

This project applies various time series forecasting techniques to predict stock market trends using real-world financial data. The models are compared based on performance metrics and visualized through an interactive Streamlit dashboard.

---

## 🧠 Project Objectives
- Understand time series concepts: trend, seasonality, noise.
- Preprocess and visualize historical stock market data.
- Implement and compare ARIMA, SARIMA, Prophet, and LSTM models.
- Evaluate performance using RMSE, MAE, and R² Score.
- Deploy results through a fully interactive dashboard.

---

## 🧰 Tech Stack & Tools
- **Python**, **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**, **Statsmodels**
- **Facebook Prophet**, **Keras/TensorFlow** (for LSTM)
- **Streamlit** (dashboard)

---

## 📦 Dataset
- Source: `AAPL.csv` (Apple stock data)
- Columns: Date, Open, High, Low, Close, Volume
- Range: 2015–2024 (daily)

---

## 📊 Models Implemented

| Model    | RMSE    | MAE     | R² Score |
|----------|---------|---------|----------|
| ARIMA    | 69.37   | 63.38   | -5.03    |
| SARIMA   | 55.78   | 51.03   | -2.90    |
| Prophet  | 38.32   | 31.40   | -0.84    |
| **LSTM** | **10.63** | **8.86** | **0.84** |

✅ **Best Model: LSTM** – Lowest RMSE and Highest R²

---

## 📉 Visual Outputs

- 📈 Actual vs Predicted chart for each model
- 📊 Comparison bar charts for RMSE, MAE, R²
- ✅ Interactive model selection in dashboard

---

StreamLit Link:- https://stockforecastingbyaryan.streamlit.app/

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

StreamLit Link:- https://stockforecastingbyaryan.streamlit.app/


