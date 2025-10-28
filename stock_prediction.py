# ================== IMPORTS ==================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Stock Price Forecasting Dashboard", layout="wide")
st.title("📈 Stock Price Prediction and Forecasting System")
st.write("Built using ARIMA | Prophet | ML | DL | Streamlit")

# ================== SIDEBAR CONFIG ==================
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. TCS.NS, INFY.NS):", "TCS.NS")
start_date = st.sidebar.text_input("Start Date", "2015-01-01")
end_date = st.sidebar.text_input("End Date", "2025-01-01")
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 30, 180, 60)

# ================== DATA FETCHING ==================
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)

    # ✅ FIX for MultiIndex columns (Yahoo Finance update)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

# ✅ Check if data fetched properly
if data.empty:
    st.error("❌ No data fetched. Please check ticker symbol or internet connection.")
    st.stop()

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA",
    "🔮 Prophet Forecast",
    "📈 ARIMA Forecast",
    "🤖 Machine Learning Models",
    "🧠 Deep Learning Models",
    "📋 Model Comparison"
])

# ================== TAB 1: EDA ==================
with tab1:
    st.subheader(f"📊 Historical Close Price for {ticker}")
    st.dataframe(data.tail())

    fig = px.line(data, x='Date', y='Close', title=f"Close Price of {ticker}")
    st.plotly_chart(fig, use_container_width=True)

    # Add moving averages
    data['SMA_7'] = data['Close'].rolling(7).mean()
    data['SMA_30'] = data['Close'].rolling(30).mean()

    st.subheader("📈 Moving Averages (SMA-7, SMA-30)")
    fig2 = px.line(data, x='Date', y=['Close', 'SMA_7', 'SMA_30'], title="Moving Averages")
    st.plotly_chart(fig2, use_container_width=True)

# ================== TAB 2: PROPHET ==================
with tab2:
    st.subheader("🔮 Prophet Forecasting")
    try:
        df_prophet = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(inplace=True)

        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=forecast_horizon)
        forecast = m.predict(future)

        st.write("📊 Prophet Forecast Data:")
        st.dataframe(forecast.tail())

        st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
        st.pyplot(m.plot_components(forecast))

    except Exception as e:
        st.error(f"⚠️ Prophet failed: {e}")

# ================== TAB 3: ARIMA ==================
with tab3:
    st.subheader("📈 ARIMA Forecasting (Statsmodels)")

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        ts = data[['Date', 'Close']].copy()
        ts.set_index('Date', inplace=True)
        ts = ts.asfreq('B').fillna(method='ffill')

        # ADF Test
        adf_result = adfuller(ts['Close'])
        st.write(f"ADF Statistic: {adf_result[0]:.3f}, P-Value: {adf_result[1]:.5f}")

        # Split
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]

        # Model
        model = ARIMA(train, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))

        # Metrics
        mae_arima = mean_absolute_error(test, forecast)
        rmse_arima = np.sqrt(mean_squared_error(test, forecast))
        r2_arima = r2_score(test, forecast)

        st.write(f"**MAE:** {mae_arima:.2f} | **RMSE:** {rmse_arima:.2f} | **R²:** {r2_arima:.3f}")

        # Plot
        fig_arima = px.line(title="ARIMA Forecast vs Actual")
        fig_arima.add_scatter(x=train.index, y=train['Close'], name='Train')
        fig_arima.add_scatter(x=test.index, y=test['Close'], name='Actual')
        fig_arima.add_scatter(x=test.index, y=forecast, name='Forecast')
        st.plotly_chart(fig_arima, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ ARIMA failed: {e}")

# ================== TAB 4: MACHINE LEARNING ==================
with tab4:
    st.subheader("🤖 Machine Learning Models (LR | RF | XGBoost)")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor

        df_ml = data.copy()
        df_ml['SMA_7'] = df_ml['Close'].rolling(7).mean()
        df_ml['SMA_30'] = df_ml['Close'].rolling(30).mean()
        df_ml['EMA'] = df_ml['Close'].ewm(span=14, adjust=False).mean()
        df_ml['Daily_Return'] = df_ml['Close'].pct_change()
        df_ml['Close_Lag1'] = df_ml['Close'].shift(1)
        df_ml['Close_Lag2'] = df_ml['Close'].shift(2)
        df_ml['Close_Lag3'] = df_ml['Close'].shift(3)
        df_ml.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Volume',
                    'SMA_7', 'SMA_30', 'EMA',
                    'Daily_Return', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3']
        X = df_ml[features]
        y = df_ml['Close']

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr = r2_score(y_test, y_pred_lr)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        r2_xgb = r2_score(y_test, y_pred_xgb)

        # Visualization
        results_df = pd.DataFrame({
            'Date': df_ml['Date'].iloc[train_size:],
            'Actual': y_test.values,
            'LR_Predicted': y_pred_lr,
            'RF_Predicted': y_pred_rf,
            'XGB_Predicted': y_pred_xgb
        })
        fig_ml = px.line(results_df, x='Date',
                         y=['Actual', 'LR_Predicted', 'RF_Predicted', 'XGB_Predicted'],
                         title="Model Predictions Comparison")
        st.plotly_chart(fig_ml, use_container_width=True)

        metrics_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'MAE': [mae_lr, mae_rf, mae_xgb],
            'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
            'R²': [r2_lr, r2_rf, r2_xgb]
        })
        st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"}))

    except Exception as e:
        st.error(f"⚠️ Machine Learning Models failed: {e}")

# ================== TAB 5: DEEP LEARNING ==================
with tab5:
    st.subheader("🤖 Deep Learning Models (LSTM & GRU)")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Prepare data
        df_dl = data[['Date', 'Close']].copy()
        df_dl['Date'] = pd.to_datetime(df_dl['Date'])
        df_dl.set_index('Date', inplace=True)

        # Scale values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_dl)

        # Sequence creation function (60 days window)
        def create_sequences(data, time_steps=60):
            X, y = [], []
            for i in range(time_steps, len(data)):
                X.append(data[i - time_steps:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        time_steps = 60
        X, y = create_sequences(scaled_data, time_steps)

        # Split train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape to 3D for neural networks
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # ========== LSTM Model ==========
        st.info("⚙️ Training LSTM Model...")
        model_lstm = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        y_pred_lstm = model_lstm.predict(X_test)
        y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_lstm)
        rmse_lstm = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_lstm))
        r2_lstm = r2_score(y_test_rescaled, y_pred_lstm)
        st.success(f"✅ LSTM → MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, R²: {r2_lstm:.3f}")

        # ========== GRU Model ==========
        st.info("⚙️ Training GRU Model...")
        model_gru = Sequential([
            GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model_gru.compile(optimizer='adam', loss='mean_squared_error')
        model_gru.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        y_pred_gru = model_gru.predict(X_test)
        y_pred_gru = scaler.inverse_transform(y_pred_gru)

        mae_gru = mean_absolute_error(y_test_rescaled, y_pred_gru)
        rmse_gru = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_gru))
        r2_gru = r2_score(y_test_rescaled, y_pred_gru)
        st.success(f"✅ GRU → MAE: {mae_gru:.2f}, RMSE: {rmse_gru:.2f}, R²: {r2_gru:.3f}")

        # ========== Visualization ==========
        pred_df = pd.DataFrame({
            'Date': df_dl.index[-len(y_test_rescaled):],
            'Actual': y_test_rescaled.flatten(),
            'LSTM_Predicted': y_pred_lstm.flatten(),
            'GRU_Predicted': y_pred_gru.flatten()
        })

        fig_dl = px.line(pred_df, x='Date', 
                         y=['Actual', 'LSTM_Predicted', 'GRU_Predicted'],
                         title="📈 LSTM vs GRU Forecasts")
        st.plotly_chart(fig_dl, use_container_width=True)

        # ========== Comparison Table ==========
        metrics_data = {
            'Model': ['LSTM', 'GRU'],
            'MAE': [mae_lstm, mae_gru],
            'RMSE': [rmse_lstm, rmse_gru],
            'R²': [r2_lstm, r2_gru]
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.write("📊 **Deep Learning Model Performance**")
        st.dataframe(metrics_df.style.format({
            "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"
        }))

    except Exception as e:
        st.error(f"⚠️ Deep Learning Models failed: {e}")

# ================== TAB 6: MODEL COMPARISON ==================
with tab6:
    st.subheader("📊 Model Comparison & Insights")

    try:
        # Combine metrics from all models
        all_results = {
            'Model': [
                'ARIMA',
                'Linear Regression', 'Random Forest', 'XGBoost',
                'LSTM', 'GRU'
            ],
            'MAE': [
                mae_arima if 'mae_arima' in locals() else np.nan,
                mae_lr if 'mae_lr' in locals() else np.nan,
                mae_rf if 'mae_rf' in locals() else np.nan,
                mae_xgb if 'mae_xgb' in locals() else np.nan,
                mae_lstm if 'mae_lstm' in locals() else np.nan,
                mae_gru if 'mae_gru' in locals() else np.nan,
            ],
            'RMSE': [
                rmse_arima if 'rmse_arima' in locals() else np.nan,
                rmse_lr if 'rmse_lr' in locals() else np.nan,
                rmse_rf if 'rmse_rf' in locals() else np.nan,
                rmse_xgb if 'rmse_xgb' in locals() else np.nan,
                rmse_lstm if 'rmse_lstm' in locals() else np.nan,
                rmse_gru if 'rmse_gru' in locals() else np.nan,
            ],
            'R²': [
                r2_arima if 'r2_arima' in locals() else np.nan,
                r2_lr if 'r2_lr' in locals() else np.nan,
                r2_rf if 'r2_rf' in locals() else np.nan,
                r2_xgb if 'r2_xgb' in locals() else np.nan,
                r2_lstm if 'r2_lstm' in locals() else np.nan,
                r2_gru if 'r2_gru' in locals() else np.nan,
            ]
        }

        results_df = pd.DataFrame(all_results)
        results_df.dropna(how='all', inplace=True)

        st.write("### 📈 Model Performance Summary")
        st.dataframe(results_df.style.format({
            "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.3f}"
        }).highlight_min(color='lightgreen', subset=['MAE', 'RMSE'])
          .highlight_max(color='lightblue', subset=['R²']))

        # Visual Comparison (RMSE)
        fig_compare = px.bar(results_df.sort_values('RMSE'),
                             x='Model', y='RMSE',
                             title="🔍 RMSE Comparison Across Models",
                             color='Model')
        st.plotly_chart(fig_compare, use_container_width=True)

        # Insights Section
        st.write("### 💡 Insights & Recommendations")

        best_model_idx = results_df['RMSE'].idxmin()
        best_model_name = results_df.loc[best_model_idx, 'Model']
        best_rmse = results_df.loc[best_model_idx, 'RMSE']
        best_r2 = results_df.loc[best_model_idx, 'R²']

        st.markdown(f"""
        ✅ **Best Performing Model:** `{best_model_name}`  
        📉 **Lowest RMSE:** `{best_rmse:.2f}`  
        📈 **R² Score:** `{best_r2:.3f}`  
        """)

        if 'LSTM' in best_model_name or 'GRU' in best_model_name:
            st.info("💬 The Deep Learning models (LSTM/GRU) tend to perform best for sequential stock data as they capture temporal dependencies effectively.")
        elif 'Prophet' in best_model_name or 'ARIMA' in best_model_name:
            st.info("💬 Time-series models like ARIMA or Prophet work well when data shows strong trend/seasonality patterns.")
        elif 'Random Forest' in best_model_name or 'XGBoost' in best_model_name:
            st.info("💬 Ensemble ML models can perform well with engineered features but may underperform on highly sequential patterns.")
        else:
            st.info("💬 Linear models give a simple baseline — good to compare but often less powerful for real stock data.")
    except Exception as e:
        st.error(f"⚠️ Model Comparison failed: {e}")