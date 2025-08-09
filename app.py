# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Stock Forecasting Dashboard")
st.markdown("Compare models and explore forecasts.")

# Simulated data for demonstration
df = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Actual': np.linspace(150, 180, 100),
})
df['ARIMA'] = df['Actual'] + np.random.normal(0, 8, 100)
df['SARIMA'] = df['Actual'] + np.random.normal(0, 6, 100)
df['Prophet'] = df['Actual'] + np.random.normal(0, 4, 100)
df['LSTM'] = df['Actual'] + np.random.normal(0, 2, 100)

# Sidebar Summary Panel
st.sidebar.header("📌 Summary Panel")
st.sidebar.metric("Best Model (RMSE)", "LSTM")
st.sidebar.metric("Lowest MAE", "LSTM")
st.sidebar.metric("Highest R² Score", "LSTM")

# Toggle for full vs recent data
st.sidebar.markdown("---")
view_option = st.sidebar.radio("View Data Range:", ["Full Dataset", "Recent 30 Days"])
df_plot = df[-30:] if view_option == "Recent 30 Days" else df

# Model Evaluation Metrics
results = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
    'RMSE': [69.3731, 55.7803, 38.3167, 10.6288],
    'MAE': [63.3772, 51.0295, 31.3954, 8.8552],
    'R²': [-5.0340, -2.9011, -0.8408, 0.8448]
})

# Model Selector
selected_model = st.selectbox("🔍 Select a model to view details", results['Model'])

# Evaluation Table
st.subheader("📊 Model Evaluation Metrics")
st.write(results[results['Model'] == selected_model].set_index('Model'))

# Forecast Plot
st.subheader(f"📉 {selected_model} Forecast - Actual vs Predicted")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_plot['Date'], df_plot['Actual'], label='Actual Price', color='blue')
ax.plot(df_plot['Date'], df_plot[selected_model], label=f'Predicted ({selected_model})', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title(f"{selected_model} - Actual vs Predicted")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Download Buttons
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 Download Forecasts (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='forecasted_data.csv',
    mime='text/csv'
)

residual_df = pd.DataFrame({
    'Date': df['Date'],
    'Residual_LSTM': df['Actual'] - df['LSTM']
})
st.sidebar.download_button(
    label="📥 Download Residuals (CSV)",
    data=residual_df.to_csv(index=False).encode('utf-8'),
    file_name='residuals_lstm.csv',
    mime='text/csv'
)

# Comparison Table
st.subheader("📌 All Model Comparison")
st.dataframe(results.set_index("Model").style
    .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen')
    .highlight_max(axis=0, subset=['R²'], color='lightgreen'))

best_model = results.loc[results['RMSE'].idxmin(), 'Model']
st.success(f"✅ Best Model Based on RMSE: **{best_model}**")

# Comparison Graph Tabs
st.subheader("📊 Comparison Graphs")
tab1, tab2, tab3 = st.tabs(["RMSE", "MAE", "R² Score"])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.barplot(data=results, x='Model', y='RMSE', ax=ax1, palette='coolwarm')
    for i, val in enumerate(results['RMSE']):
        ax1.text(i, val + 2, f"{val:.2f}", ha='center')
    ax1.set_title("RMSE Comparison")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.barplot(data=results, x='Model', y='MAE', ax=ax2, palette='crest')
    for i, val in enumerate(results['MAE']):
        ax2.text(i, val + 2, f"{val:.2f}", ha='center')
    ax2.set_title("MAE Comparison")
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    sns.barplot(data=results, x='Model', y='R²', ax=ax3, palette='magma')
    for i, val in enumerate(results['R²']):
        ax3.text(i, val + 0.1, f"{val:.2f}", ha='center')
    ax3.set_title("R² Score Comparison")
    st.pyplot(fig3)

# LSTM Diagnostics
st.subheader("🔍 LSTM Residual Diagnostics")
residuals = df['Actual'] - df['LSTM']

fig4, ax4 = plt.subplots()
sns.histplot(residuals, bins=30, kde=True, ax=ax4, color='purple')
ax4.set_title('LSTM Residuals Distribution')
ax4.set_xlabel('Prediction Error')
st.pyplot(fig4)

fig5, ax5 = plt.subplots()
ax5.scatter(df['Actual'], residuals, alpha=0.6, color='orange')
ax5.axhline(0, linestyle='--', color='gray')
ax5.set_title('LSTM Residuals vs Actual Values')
ax5.set_xlabel('Actual Stock Price')
ax5.set_ylabel('Residuals')
st.pyplot(fig5)

fig6, ax6 = plt.subplots()
ax6.plot(df['Date'][-30:], df['Actual'][-30:], label='Actual', linewidth=2)
ax6.plot(df['Date'][-30:], df['LSTM'][-30:], label='LSTM Forecast', linestyle='--')
ax6.set_title("LSTM Forecast vs Actual (Last 30 Days)")
ax6.set_xlabel('Date')
ax6.set_ylabel('Stock Price')
ax6.legend()
st.pyplot(fig6)

# Residual Boxplot for All Models
fig7, ax7 = plt.subplots()
res_df = pd.DataFrame({
    'ARIMA': df['Actual'] - df['ARIMA'],
    'SARIMA': df['Actual'] - df['SARIMA'],
    'Prophet': df['Actual'] - df['Prophet'],
    'LSTM': df['Actual'] - df['LSTM']
})
sns.boxplot(data=res_df, ax=ax7)
ax7.set_title("Model Error Comparison (Residual Boxplot)")
ax7.set_ylabel("Residuals")
st.pyplot(fig7)

# Q-Q Plot
fig8 = plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of LSTM Residuals")
plt.grid(True)
st.pyplot(fig8)

# ✅ NEW Residual Distribution Section for All Models
st.subheader("📊 Residual Distribution for All Models")
fig9, axes = plt.subplots(2, 2, figsize=(12, 8))
models = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM']
for i, model in enumerate(models):
    row, col = divmod(i, 2)
    sns.histplot(df['Actual'] - df[model], bins=30, kde=True, ax=axes[row][col])
    axes[row][col].set_title(f'{model} Residual Distribution')
    axes[row][col].set_xlabel('Residuals')
plt.tight_layout()
st.pyplot(fig9)


