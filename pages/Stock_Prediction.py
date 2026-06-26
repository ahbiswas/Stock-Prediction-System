import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_stock_prediction():
    st.title("🤖 Stock Price Prediction - Comparative Model Analysis")
    
    st.markdown("""
    ### Compare LSTM, GRU, and ARIMA Models
    
    This module provides AI-powered stock price forecasting using three different approaches:
    - **LSTM**: Long Short-Term Memory neural network for sequential data
    - **GRU**: Gated Recurrent Unit neural network (faster than LSTM)
    - **ARIMA**: AutoRegressive Integrated Moving Average (statistical approach)
    
    Compare the performance of each model and get 30-day price predictions.
    """)
    
    st.markdown("---")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Model Parameters")
        
        companies = {
            "Tesla (TSLA)": "TSLA",
            "Apple (AAPL)": "AAPL",
            "Alphabet Inc. (GOOGL)": "GOOGL",
            "Amazon (AMZN)": "AMZN",
            "Microsoft (MSFT)": "MSFT",
            "Meta Platforms (META)": "META",
            "NVIDIA (NVDA)": "NVDA",
            "Netflix (NFLX)": "NFLX"
        }
        
        selected_company = st.selectbox("Select Company", list(companies.keys()), index=0)
        ticker = companies[selected_company]
        
        today = dt.date.today()
        start_date = st.date_input("Start Date", dt.date(today.year - 3, today.month, today.day))
        end_date = st.date_input("End Date", today)
        
        st.markdown("---")
        st.subheader("🔧 Model Settings")
        
        test_size = st.slider("Test Data Size (%)", min_value=10, max_value=30, value=20, step=5)
        lookback_days = st.selectbox("Lookback Days", [30, 60, 90, 120], index=1)
        epochs = st.selectbox("Training Epochs", [50, 100, 150, 200], index=2)
        lstm_units = st.selectbox("LSTM/GRU Units", [50, 100, 150, 200], index=1)
        
        st.markdown("---")
        st.subheader("📊 ARIMA Parameters")
        arima_p = st.selectbox("p (AR order)", [1, 2, 3, 4, 5], index=1)
        arima_d = st.selectbox("d (Differencing)", [0, 1, 2], index=0)
        arima_q = st.selectbox("q (MA order)", [1, 2, 3, 4, 5], index=0)
    
    @st.cache_data
    def fetch_data(ticker, start, end):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        return data
    
    def create_sequences(data, lookback=60):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_lstm_model(units, lookback):
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(units//2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def build_gru_model(units, lookback):
        model = Sequential([
            GRU(units, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            GRU(units//2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_predict_models(data, lookback, test_size, epochs, units):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['Close']].values)
        
        train_size = int(len(scaled_data) * (1 - test_size/100))
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - lookback:]
        
        X_train, y_train = create_sequences(train_data, lookback)
        X_test, y_test = create_sequences(test_data, lookback)
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        with st.spinner("Training LSTM model..."):
            lstm_model = build_lstm_model(units, lookback)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
            lstm_pred = lstm_model.predict(X_test, verbose=0)
        
        with st.spinner("Training GRU model..."):
            gru_model = build_gru_model(units, lookback)
            gru_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
            gru_pred = gru_model.predict(X_test, verbose=0)
        
        with st.spinner("Training ARIMA model..."):
            arima_model = ARIMA(data['Close'].values[:train_size], order=(arima_p, arima_d, arima_q))
            arima_fitted = arima_model.fit()
            arima_pred = arima_fitted.forecast(len(y_test))
        
        lstm_pred = scaler.inverse_transform(lstm_pred)
        gru_pred = scaler.inverse_transform(gru_pred)
        actual_values = data['Close'].values[train_size:]
        
        return {
            'lstm': lstm_pred.flatten(),
            'gru': gru_pred.flatten(),
            'arima': arima_pred,
            'actual': actual_values,
            'scaler': scaler,
            'train_size': train_size,
            'data': data,
            'last_sequence': scaled_data[-lookback:]
        }
    
    def predict_future(model, last_sequence, scaler, days=30):
        predictions = []
        current_seq = last_sequence.copy()
        for _ in range(days):
            next_pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
            predictions.append(next_pred[0, 0])
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = next_pred[0, 0]
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    if st.button("🚀 Run Predictions", type="primary"):
        data = fetch_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error("No data available for the selected date range.")
            st.stop()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Days", len(data))
        with col2:
            st.metric("📈 Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col3:
            st.metric("📉 Min Price", f"${data['Close'].min():.2f}")
        with col4:
            st.metric("📈 Max Price", f"${data['Close'].max():.2f}")
        
        results = train_predict_models(data, lookback_days, test_size, epochs, lstm_units)
        
        metrics = {}
        for model_name in ['lstm', 'gru', 'arima']:
            pred = results[model_name]
            actual = results['actual'][-len(pred):]
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            metrics[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
        
        st.markdown("## 📊 Model Performance Comparison")
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', subset=['R²']), use_container_width=True)
        
        best_model = min(metrics, key=lambda x: metrics[x]['RMSE'])
        st.success(f"🏆 Best Performing Model: **{best_model.upper()}** (RMSE: {metrics[best_model]['RMSE']:.4f})")
        
        st.markdown("## 📈 Model Predictions vs Actual Values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[results['train_size']:], y=results['actual'], mode='lines', name='Actual', line=dict(color='black', width=2)))
        
        colors = {'lstm': 'blue', 'gru': 'green', 'arima': 'red'}
        for model_name in ['lstm', 'gru', 'arima']:
            pred_values = results[model_name]
            fig.add_trace(go.Scatter(x=data.index[results['train_size']:results['train_size'] + len(pred_values)], y=pred_values, mode='lines', name=f'{model_name.upper()} Prediction', line=dict(color=colors[model_name], width=1.5, dash='dash')))
        
        fig.update_layout(title=f'{selected_company} - Model Predictions vs Actual', xaxis_title='Date', yaxis_title='Price ($)', height=600, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("## 🔮 30-Day Price Predictions")
        
        with st.spinner("Generating 30-day predictions..."):
            scaler = MinMaxScaler()
            scaled_full = scaler.fit_transform(data[['Close']].values)
            
            X_full, y_full = create_sequences(scaled_full, lookback_days)
            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
            
            lstm_model = build_lstm_model(lstm_units, lookback_days)
            lstm_model.fit(X_full, y_full, epochs=epochs, batch_size=32, verbose=0)
            
            gru_model = build_gru_model(lstm_units, lookback_days)
            gru_model.fit(X_full, y_full, epochs=epochs, batch_size=32, verbose=0)
            
            arima_model_full = ARIMA(data['Close'].values, order=(arima_p, arima_d, arima_q))
            arima_fitted_full = arima_model_full.fit()
            
            last_seq = scaled_full[-lookback_days:]
            lstm_future = predict_future(lstm_model, last_seq, scaler, 30)
            gru_future = predict_future(gru_model, last_seq, scaler, 30)
            arima_future = arima_fitted_full.forecast(30)
            
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=30)
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'LSTM': lstm_future,
            'GRU': gru_future,
            'ARIMA': arima_future
        })
        future_df['Average'] = future_df[['LSTM', 'GRU', 'ARIMA']].mean(axis=1)
        
        st.dataframe(future_df.style.format({'LSTM': '${:.2f}', 'GRU': '${:.2f}', 'ARIMA': '${:.2f}', 'Average': '${:.2f}'}), use_container_width=True)
        
        st.markdown("### 📈 Future Price Trends")
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=data.index[-60:], y=data['Close'].values[-60:], mode='lines', name='Historical', line=dict(color='black', width=2)))
        
        colors = {'LSTM': 'blue', 'GRU': 'green', 'ARIMA': 'red', 'Average': 'purple'}
        for model in ['LSTM', 'GRU', 'ARIMA', 'Average']:
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_df[model], mode='lines+markers', name=model, line=dict(color=colors[model], width=2, dash='dot'), marker=dict(size=6)))
        
        fig_future.update_layout(title=f'{selected_company} - 30-Day Price Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=600, hovermode='x unified')
        st.plotly_chart(fig_future, use_container_width=True)
        
        st.markdown("### 📋 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 LSTM Predicted Price", f"${lstm_future[-1]:.2f}", f"{((lstm_future[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
        with col2:
            st.metric("📈 GRU Predicted Price", f"${gru_future[-1]:.2f}", f"{((gru_future[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
        with col3:
            st.metric("📈 ARIMA Predicted Price", f"${arima_future[-1]:.2f}", f"{((arima_future[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%")
        
        csv = future_df.to_csv(index=False)
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name=f"{ticker}_predictions_30days.csv", mime="text/csv")
    else:
        st.info("👈 Adjust parameters in the sidebar and click 'Run Predictions' to start the analysis.")
        st.markdown("""
        ### 🎯 How It Works
        **📊 Model Comparison**
        - **LSTM**: Best for long-term dependencies
        - **GRU**: Faster training, similar performance
        - **ARIMA**: Traditional statistical approach
        
        **📈 Evaluation Metrics**
        - MSE, RMSE, MAE, R²
        
        **🔮 30-Day Forecast**
        - Historical data analysis, Pattern recognition, Confidence intervals
        """)

if __name__ == "__main__":
    run_stock_prediction()

# # Stock_Prediction.py - Fixed Version
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import datetime as dt
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import warnings
# warnings.filterwarnings('ignore')

# # Deep Learning imports
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# # STATSMODELS - Fix ARIMA import
# try:
#     from statsmodels.tsa.arima.model import ARIMA
#     from statsmodels.tsa.stattools import adfuller
#     from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#     import statsmodels.api as sm
# except ImportError:
#     st.error("Please install statsmodels: pip install statsmodels")
#     st.stop()

# def run_stock_prediction():
#     st.title("🤖 Stock Price Prediction - Comparative Model Analysis")
    
#     st.markdown("""
#     ### Compare LSTM, GRU, and ARIMA Models
    
#     This module provides AI-powered stock price forecasting using three different approaches:
#      - **LSTM**: Long Short-Term Memory neural network for sequential data
#      - **GRU**: Gated Recurrent Unit neural network (faster than LSTM)
#      - **ARIMA**: AutoRegressive Integrated Moving Average (statistical approach)
    
#       Compare the performance of each model and get 30-day price predictions.
#     """)
    
#     st.markdown("---")
    
#     # Sidebar for parameters
#     with st.sidebar:
#         st.header("⚙️ Model Parameters")
        
#         companies = {
#             "Tesla (TSLA)": "TSLA",
#             "Apple (AAPL)": "AAPL",
#             "Alphabet Inc. (GOOGL)": "GOOGL",
#             "Amazon (AMZN)": "AMZN",
#             "Microsoft (MSFT)": "MSFT",
#             "Meta Platforms (META)": "META",
#             "NVIDIA (NVDA)": "NVDA",
#             "Netflix (NFLX)": "NFLX",
#             "Adobe (ADBE)": "ADBE",
#             "Salesforce (CRM)": "CRM"
#         }
        
#         selected_company = st.selectbox("Select Company", list(companies.keys()), index=0)
#         ticker = companies[selected_company]
        
#         today = dt.date.today()
#         start_date = st.date_input("Start Date", dt.date(today.year - 3, today.month, today.day))
#         end_date = st.date_input("End Date", today)
        
#         st.markdown("---")
#         st.subheader("🔧 Model Settings")
        
#         test_size = st.slider("Test Data Size (%)", min_value=10, max_value=30, value=20, step=5)
#         lookback_days = st.selectbox("Lookback Days (LSTM/GRU)", [30, 60, 90, 120], index=1)
#         epochs = st.selectbox("Training Epochs", [50, 100, 150, 200], index=2)
#         lstm_units = st.selectbox("LSTM/GRU Units", [50, 100, 150, 200], index=1)
        
#         st.markdown("---")
#         st.subheader("📊 ARIMA Settings (Auto-Optimized)")
        
#         # Allow user to choose auto or manual
#         arima_mode = st.radio("ARIMA Mode", ["Auto (Recommended)", "Manual"], index=0)
        
#         if arima_mode == "Manual":
#             arima_p = st.selectbox("p (AR order)", [0, 1, 2, 3, 4, 5], index=1)
#             arima_d = st.selectbox("d (Differencing)", [0, 1, 2], index=1)
#             arima_q = st.selectbox("q (MA order)", [0, 1, 2, 3, 4, 5], index=0)
#         else:
#             arima_p, arima_d, arima_q = 1,  # Default, will be auto-optimized
    
#     # ==================== DATA FETCHING ====================
#     @st.cache_data
#     def fetch_data(ticker, start, end):
#         try:
#             stock = yf.Ticker(ticker)
#             data = stock.history(start=start, end=end)
            
#             # Ensure we have data
#             if data.empty:
#                 return data
            
#             # Add returns for ARIMA
#             data['Returns'] = data['Close'].pct_change()
#             data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
#             return data
#         except Exception as e:
#             st.error(f"Error fetching data: {e}")
#             return pd.DataFrame()
    
#     # ==================== ARIMA HELPER FUNCTIONS ====================
    
#     def check_stationarity(timeseries):
#         """Check if time series is stationary using ADF test"""
#         try:
#             result = adfuller(timeseries.dropna())
#             p_value = result[1]
#             return p_value < 0.05  # True if stationary
#         except:
#             return False
    
#     def find_best_arima_order(data, max_p=5, max_d=2, max_q=5):
#         """Auto-find best ARIMA parameters"""
#         best_aic = np.inf
#         best_order = (2, 1, 2)
        
#         # Try different combinations
#         for p in range(0, max_p + 1):
#             for d in range(0, max_d + 1):
#                 for q in range(0, max_q + 1):
#                     try:
#                         model = ARIMA(data, order=(p, d, q))
#                         model_fit = model.fit()
#                         if model_fit.aic < best_aic:
#                             best_aic = model_fit.aic
#                             best_order = (p, d, q)
#                     except:
#                         continue
        
#         return best_order
    
#     # ==================== SEQUENCE CREATION ====================
    
#     def create_sequences(data, lookback=60):
#         X, y = [], []
#         for i in range(lookback, len(data)):
#             X.append(data[i-lookback:i, 0])
#             y.append(data[i, 0])
#         return np.array(X), np.array(y)
    
#     # ==================== LSTM/GRU MODELS ====================
    
#     def build_lstm_model(units, lookback):
#         model = Sequential([
#             LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
#             Dropout(0.2),
#             LSTM(units//2, return_sequences=False),
#             Dropout(0.2),
#             Dense(25, activation='relu'),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         return model
    
#     def build_gru_model(units, lookback):
#         model = Sequential([
#             GRU(units, return_sequences=True, input_shape=(lookback, 1)),
#             Dropout(0.2),
#             GRU(units//2, return_sequences=False),
#             Dropout(0.2),
#             Dense(25, activation='relu'),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         return model
    
#     # ==================== MAIN TRAINING FUNCTION ====================
    
#     def train_predict_models(data, lookback, test_size, epochs, units, arima_mode, arima_params):
        
#         # Prepare data
#         close_prices = data[['Close']].values
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(close_prices)
        
#         # Split data
#         train_size = int(len(scaled_data) * (1 - test_size/100))
        
#         # For ARIMA - use original prices
#         train_prices = close_prices[:train_size].flatten()
#         test_prices = close_prices[train_size:].flatten()
        
#         # ========== LSTM ==========
#         with st.spinner("Training LSTM model..."):
#             train_data = scaled_data[:train_size]
#             test_data = scaled_data[train_size - lookback:]
            
#             X_train, y_train = create_sequences(train_data, lookback)
#             X_test, y_test = create_sequences(test_data, lookback)
            
#             X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#             X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
#             lstm_model = build_lstm_model(units, lookback)
#             early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
#             lstm_model.fit(
#                 X_train, y_train,
#                 epochs=epochs,
#                 batch_size=32,
#                 validation_split=0.1,
#                 callbacks=[early_stop],
#                 verbose=0
#             )
            
#             lstm_pred = lstm_model.predict(X_test, verbose=0)
#             lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
        
#         # ========== GRU ==========
#         with st.spinner("Training GRU model..."):
#             gru_model = build_gru_model(units, lookback)
#             gru_model.fit(
#                 X_train, y_train,
#                 epochs=epochs,
#                 batch_size=32,
#                 validation_split=0.1,
#                 callbacks=[early_stop],
#                 verbose=0
#             )
            
#             gru_pred = gru_model.predict(X_test, verbose=0)
#             gru_pred = scaler.inverse_transform(gru_pred).flatten()
        
#         # ========== ARIMA ==========
#         with st.spinner("Training ARIMA model with auto-optimization..."):
#             try:
#                 # Get the actual test length
#                 test_len = len(test_prices)
                
#                 if arima_mode == "Auto (Recommended)":
#                     # Auto-find best parameters
#                     st.info("🔍 Auto-optimizing ARIMA parameters... This may take a moment.")
#                     best_order = find_best_arima_order(train_prices)
#                     p, d, q = best_order
#                     st.success(f"✅ Best ARIMA order found: ({p}, {d}, {q})")
#                 else:
#                     p, d, q = arima_params
                
#                 # Fit ARIMA model
#                 arima_model = ARIMA(train_prices, order=(p, d, q))
#                 arima_fitted = arima_model.fit()
                
#                 # Make predictions
#                 arima_pred = arima_fitted.forecast(steps=test_len)
                
#                 # If predictions are shorter than test_len, pad with last value
#                 if len(arima_pred) < test_len:
#                     last_value = arima_pred[-1] if len(arima_pred) > 0 else train_prices[-1]
#                     padding = [last_value] * (test_len - len(arima_pred))
#                     arima_pred = np.concatenate([arima_pred, padding])
#                 elif len(arima_pred) > test_len:
#                     arima_pred = arima_pred[:test_len]
                
#                 arima_pred = np.array(arima_pred).flatten()
                
#             except Exception as e:
#                 st.warning(f"⚠️ ARIMA error: {e}. Using simple moving average fallback.")
#                 # Fallback to simple moving average
#                 window = min(30, len(train_prices))
#                 arima_pred = np.array([np.mean(train_prices[-window:])] * test_len)
        
#         # Get actual values for test set
#         actual_values = test_prices
        
#         # Ensure all predictions have same length
#         min_len = min(len(actual_values), len(lstm_pred), len(gru_pred), len(arima_pred))
#         actual_values = actual_values[:min_len]
#         lstm_pred = lstm_pred[:min_len]
#         gru_pred = gru_pred[:min_len]
#         arima_pred = arima_pred[:min_len]
        
#         return {
#             'lstm': lstm_pred,
#             'gru': gru_pred,
#             'arima': arima_pred,
#             'actual': actual_values,
#             'scaler': scaler,
#             'train_size': train_size,
#             'data': data,
#             'last_sequence': scaled_data[-lookback:]
#         }
    
#     # ==================== FUTURE PREDICTIONS ====================
    
#     def predict_future(model, last_sequence, scaler, days=30):
#         predictions = []
#         current_seq = last_sequence.copy()
        
#         for _ in range(days):
#             try:
#                 next_pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
#                 predictions.append(next_pred[0, 0])
#                 current_seq = np.roll(current_seq, -1)
#                 current_seq[-1] = next_pred[0, 0]
#             except:
#                 predictions.append(current_seq[-1])
#                 current_seq = np.roll(current_seq, -1)
#                 current_seq[-1] = current_seq[-1]
        
#         try:
#             return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
#         except:
#             return np.array(predictions)
    
#     # ==================== MAIN EXECUTION ====================
    
#     if st.button("🚀 Run Predictions", type="primary"):
        
#         # Fetch data
#         with st.spinner("Fetching stock data..."):
#             data = fetch_data(ticker, start_date, end_date)
        
#         if data.empty:
#             st.error("❌ No data available for the selected date range. Please try different dates.")
#             st.stop()
        
#         # Display data info
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("📊 Total Days", len(data))
#         with col2:
#             st.metric("📈 Current Price", f"${data['Close'].iloc[-1]:.2f}")
#         with col3:
#             st.metric("📉 Min Price", f"${data['Close'].min():.2f}")
#         with col4:
#             st.metric("📈 Max Price", f"${data['Close'].max():.2f}")
        
#         # Train models
#         arima_params = (arima_p, arima_d, arima_q) if 'arima_mode' in locals() and arima_mode == "Manual" else (2, 1, 2)
#         results = train_predict_models(
#             data, lookback_days, test_size, epochs, lstm_units, 
#             arima_mode if 'arima_mode' in locals() else "Auto (Recommended)",
#             arima_params
#         )
        
#         # ========== METRICS ==========
#         metrics = {}
#         for model_name in ['lstm', 'gru', 'arima']:
#             pred = results[model_name]
#             actual = results['actual'][-len(pred):]
            
#             # Handle NaN or inf values
#             pred = np.nan_to_num(pred)
#             actual = np.nan_to_num(actual)
            
#             if len(pred) > 0 and len(actual) > 0:
#                 mse = mean_squared_error(actual, pred)
#                 rmse = np.sqrt(mse)
#                 mae = mean_absolute_error(actual, pred)
#                 r2 = r2_score(actual, pred) if len(actual) > 1 else 0
                
#                 metrics[model_name] = {
#                     'MSE': mse,
#                     'RMSE': rmse,
#                     'MAE': mae,
#                     'R²': max(0, r2)  # R² can be negative, clamp to 0
#                 }
#             else:
#                 metrics[model_name] = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'R²': 0}
        
#         # ========== DISPLAY METRICS ==========
#         st.markdown("## 📊 Model Performance Comparison")
        
#         metrics_df = pd.DataFrame(metrics).T
#         st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', subset=['R²']), use_container_width=True)
        
#         # Best model
#         if metrics:
#             best_model = min(metrics, key=lambda x: metrics[x]['RMSE'])
#             st.success(f"🏆 Best Performing Model: **{best_model.upper()}** (RMSE: {metrics[best_model]['RMSE']:.4f})")
        
#         # ========== PREDICTION VS ACTUAL PLOT ==========
#         st.markdown("## 📈 Model Predictions vs Actual Values")
        
#         fig = go.Figure()
        
#         # Actual values
#         actual_dates = data.index[results['train_size']:results['train_size'] + len(results['actual'])]
#         fig.add_trace(go.Scatter(
#             x=actual_dates,
#             y=results['actual'],
#             mode='lines',
#             name='Actual',
#             line=dict(color='black', width=3)
#         ))
        
#         # Predictions
#         colors = {'lstm': 'blue', 'gru': 'green', 'arima': 'red'}
#         for model_name in ['lstm', 'gru', 'arima']:
#             pred_values = results[model_name]
#             pred_dates = actual_dates[:len(pred_values)]
#             fig.add_trace(go.Scatter(
#                 x=pred_dates,
#                 y=pred_values,
#                 mode='lines',
#                 name=f'{model_name.upper()} Prediction',
#                 line=dict(color=colors[model_name], width=1.5, dash='dash')
#             ))
        
#         fig.update_layout(
#             title=f'{selected_company} - Model Predictions vs Actual',
#             xaxis_title='Date',
#             yaxis_title='Price ($)',
#             height=600,
#             hovermode='x unified',
#             legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # ========== 30-DAY FUTURE PREDICTIONS ==========
#         st.markdown("## 🔮 30-Day Price Predictions")
        
#         with st.spinner("Generating 30-day predictions..."):
#             try:
#                 # Train models on full data
#                 scaler = MinMaxScaler()
#                 close_prices = data[['Close']].values
#                 scaled_full = scaler.fit_transform(close_prices)
                
#                 X_full, y_full = create_sequences(scaled_full, lookback_days)
#                 X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
                
#                 # LSTM
#                 lstm_model = build_lstm_model(lstm_units, lookback_days)
#                 lstm_model.fit(X_full, y_full, epochs=epochs, batch_size=32, verbose=0)
                
#                 # GRU
#                 gru_model = build_gru_model(lstm_units, lookback_days)
#                 gru_model.fit(X_full, y_full, epochs=epochs, batch_size=32, verbose=0)
                
#                 # ARIMA on full data
#                 try:
#                     arima_model_full = ARIMA(data['Close'].values, order=(2, 1, 2))
#                     arima_fitted_full = arima_model_full.fit()
#                     arima_future = arima_fitted_full.forecast(steps=30)
                    
#                     # Handle if ARIMA returns fewer predictions
#                     if len(arima_future) < 30:
#                         last_val = arima_future[-1] if len(arima_future) > 0 else data['Close'].iloc[-1]
#                         arima_future = np.concatenate([arima_future, [last_val] * (30 - len(arima_future))])
#                 except:
#                     arima_future = np.array([data['Close'].iloc[-1]] * 30)
                
#                 # Get LSTM/GRU future predictions
#                 last_seq = scaled_full[-lookback_days:]
#                 lstm_future = predict_future(lstm_model, last_seq, scaler, 30)
#                 gru_future = predict_future(gru_model, last_seq, scaler, 30)
                
#                 # Create future dates
#                 last_date = data.index[-1]
#                 future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=30)
                
#             except Exception as e:
#                 st.error(f"Error generating future predictions: {e}")
#                 st.stop()
        
#         # Create DataFrame
#         future_df = pd.DataFrame({
#             'Date': future_dates,
#             'LSTM': lstm_future,
#             'GRU': gru_future,
#             'ARIMA': arima_future[:30]
#         })
        
#         # Ensure all columns have same length
#         future_df['ARIMA'] = future_df['ARIMA'].fillna(future_df['LSTM'])
#         future_df['Average'] = future_df[['LSTM', 'GRU', 'ARIMA']].mean(axis=1)
        
#         # Display predictions
#         st.dataframe(
#             future_df.style.format({
#                 'LSTM': '${:.2f}',
#                 'GRU': '${:.2f}',
#                 'ARIMA': '${:.2f}',
#                 'Average': '${:.2f}'
#             }),
#             use_container_width=True
#         )
        
#         # ========== FUTURE TREND PLOT ==========
#         st.markdown("### 📈 Future Price Trends")
        
#         fig_future = go.Figure()
        
#         # Historical data (last 60 days)
#         hist_data = data.iloc[-60:]
#         fig_future.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['Close'].values,
#             mode='lines',
#             name='Historical',
#             line=dict(color='black', width=2)
#         ))
        
#         # Future predictions
#         colors = {'LSTM': 'blue', 'GRU': 'green', 'ARIMA': 'red', 'Average': 'purple'}
#         for model in ['LSTM', 'GRU', 'ARIMA', 'Average']:
#             fig_future.add_trace(go.Scatter(
#                 x=future_dates,
#                 y=future_df[model],
#                 mode='lines+markers',
#                 name=model,
#                 line=dict(color=colors[model], width=2, dash='dot'),
#                 marker=dict(size=6)
#             ))
        
#         fig_future.update_layout(
#             title=f'{selected_company} - 30-Day Price Forecast',
#             xaxis_title='Date',
#             yaxis_title='Price ($)',
#             height=600,
#             hovermode='x unified',
#             legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
#         )
        
#         st.plotly_chart(fig_future, use_container_width=True)
        
#         # ========== PREDICTION SUMMARY ==========
#         st.markdown("### 📋 Prediction Summary")
        
#         col1, col2, col3 = st.columns(3)
#         current_price = data['Close'].iloc[-1]
        
#         with col1:
#             lstm_change = ((lstm_future[-1] - current_price) / current_price * 100)
#             st.metric(
#                 "📈 LSTM Predicted Price",
#                 f"${lstm_future[-1]:.2f}",
#                 f"{lstm_change:.2f}%",
#                 delta_color="normal" if lstm_change > 0 else "inverse"
#             )
#         with col2:
#             gru_change = ((gru_future[-1] - current_price) / current_price * 100)
#             st.metric(
#                 "📈 GRU Predicted Price",
#                 f"${gru_future[-1]:.2f}",
#                 f"{gru_change:.2f}%",
#                 delta_color="normal" if gru_change > 0 else "inverse"
#             )
#         with col3:
#             arima_change = ((arima_future[-1] - current_price) / current_price * 100)
#             st.metric(
#                 "📈 ARIMA Predicted Price",
#                 f"${arima_future[-1]:.2f}",
#                 f"{arima_change:.2f}%",
#                 delta_color="normal" if arima_change > 0 else "inverse"
#             )
        
#         # ========== DOWNLOAD BUTTON ==========
#         csv = future_df.to_csv(index=False)
#         st.download_button(
#             "📥 Download Predictions as CSV",
#             data=csv,
#             file_name=f"{ticker}_predictions_30days.csv",
#             mime="text/csv"
#         )
        
#         # Show ARIMA parameters used
#         st.info(f"📊 ARIMA order used: ({arima_p if 'arima_mode' in locals() and arima_mode == 'Manual' else 'auto-optimized'})")
    
#     else:
#         st.info("👈 Adjust parameters in the sidebar and click 'Run Predictions' to start the analysis.")
        
#         # ========== HOW IT WORKS SECTION ==========
#         st.markdown("### 🎯 How It Works")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("""
#             **📊 Model Comparison**
#             - **LSTM**: Best for long-term dependencies
#             - **GRU**: Faster training, similar performance
#             - **ARIMA**: Traditional statistical approach (Now Fixed!)
            
#             **📈 Evaluation Metrics**
#             - MSE (Mean Squared Error)
#             - RMSE (Root Mean Squared Error)
#             - MAE (Mean Absolute Error)
#             - R² (Coefficient of Determination)
#             """)
#         with col2:
#             st.markdown("""
#             **🔮 30-Day Forecast**
#             - Historical data analysis
#             - Pattern recognition
#             - Trend extrapolation
#             - Confidence intervals
            
#             **📋 Features**
#             - Interactive visualizations
#             - Performance comparison
#             - Downloadable results
#             - Model transparency
#             """)

# if __name__ == "__main__":
#     run_stock_prediction()



# # Stock_Prediction.py - Complete Auto-ARIMA Implementation
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import datetime as dt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import warnings
# warnings.filterwarnings('ignore')

# # Deep Learning imports
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# # Statsmodels for ARIMA
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import matplotlib.pyplot as plt

# # Auto-ARIMA with pmdarima
# try:
#     from pmdarima import auto_arima
#     from pmdarima.arima import ADFTest, KPSSTest
# except ImportError:
#     st.error("Please install pmdarima: pip install pmdarima")
#     st.stop()

# def run_stock_prediction():
#     st.set_page_config(
#         page_title="Stock Prediction - Auto ARIMA",
#         page_icon="🤖",
#         layout="wide"
#     )
    
#     st.title("🤖 Stock Price Prediction - Auto-ARIMA Powered")
    
#     st.markdown("""
#     ### 📊 Advanced Time Series Forecasting with Auto-ARIMA
    
#     **Auto-ARIMA automatically finds the optimal parameters for your data!**
#     - 🔍 **Automated Stationarity Testing** (ADF, KPSS)
#     - 📈 **Smart Parameter Selection** (AIC/BIC optimization)
#     - ⚡ **Fast & Accurate** Predictions
#     - 🎯 **Handles Seasonality** Automatically
#     """)
    
#     st.markdown("---")
    
#     # ==================== SIDEBAR ====================
#     with st.sidebar:
#         st.header("⚙️ Model Parameters")
        
#         companies = {
#             "Tesla (TSLA)": "TSLA",
#             "Apple (AAPL)": "AAPL",
#             "Alphabet Inc. (GOOGL)": "GOOGL",
#             "Amazon (AMZN)": "AMZN",
#             "Microsoft (MSFT)": "MSFT",
#             "Meta Platforms (META)": "META",
#             "NVIDIA (NVDA)": "NVDA",
#             "Netflix (NFLX)": "NFLX",
#             "Adobe (ADBE)": "ADBE",
#             "Salesforce (CRM)": "CRM",
#             "Intel (INTC)": "INTC",
#             "IBM (IBM)": "IBM"
#         }
        
#         selected_company = st.selectbox("Select Company", list(companies.keys()), index=0)
#         ticker = companies[selected_company]
        
#         today = dt.date.today()
#         start_date = st.date_input("Start Date", dt.date(today.year - 3, today.month, today.day))
#         end_date = st.date_input("End Date", today)
        
#         st.markdown("---")
#         st.subheader("🔧 Model Settings")
        
#         test_size = st.slider("Test Data Size (%)", min_value=10, max_value=30, value=20, step=5)
#         lookback_days = st.selectbox("Lookback Days (LSTM/GRU)", [30, 60, 90, 120], index=1)
#         epochs = st.selectbox("Training Epochs", [50, 100, 150, 200], index=2)
#         lstm_units = st.selectbox("LSTM/GRU Units", [50, 100, 150, 200], index=1)
        
#         st.markdown("---")
#         st.subheader("📊 Auto-ARIMA Configuration")
        
#         # Auto-ARIMA parameters
#         seasonal = st.checkbox("Enable Seasonal", value=True)
#         if seasonal:
#             m = st.selectbox("Seasonal Period (m)", [7, 12, 24, 52], index=0, 
#                            help="7=daily, 12=monthly, 52=weekly for daily data")
#         else:
#             m = 1
        
#         max_p = st.slider("Max p (AR order)", 1, 10, 5)
#         max_d = st.slider("Max d (Differencing)", 0, 3, 2)
#         max_q = st.slider("Max q (MA order)", 1, 10, 5)
        
#         information_criterion = st.selectbox("Information Criterion", ['aic', 'bic', 'hqic'], index=0)
        
#         stepwise = st.checkbox("Stepwise (Faster)", value=True)
#         trace = st.checkbox("Show Trace (Debug)", value=False)
    
#     # ==================== DATA FETCHING ====================
#     @st.cache_data
#     def fetch_data(ticker, start, end):
#         try:
#             stock = yf.Ticker(ticker)
#             data = stock.history(start=start, end=end)
            
#             if data.empty:
#                 return data
            
#             # Basic preprocessing
#             data['Returns'] = data['Close'].pct_change()
#             data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
#             data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
#             return data
#         except Exception as e:
#             st.error(f"Error fetching data: {e}")
#             return pd.DataFrame()
    
#     # ==================== AUTO-ARIMA FUNCTIONS ====================
    
#     def check_stationarity(series, title='Time Series'):
#         """Perform ADF and KPSS tests for stationarity"""
#         results = {}
        
#         # ADF Test
#         adf_test = ADFTest(alpha=0.05)
#         is_stationary, p_value = adf_test.is_stationary(series.dropna())
#         results['ADF'] = {
#             'is_stationary': is_stationary,
#             'p_value': p_value,
#             'test': 'Augmented Dickey-Fuller'
#         }
        
#         # KPSS Test
#         kpss_test = KPSSTest(alpha=0.05)
#         is_stationary_kpss, p_value_kpss = kpss_test.is_stationary(series.dropna())
#         results['KPSS'] = {
#             'is_stationary': is_stationary_kpss,
#             'p_value': p_value_kpss,
#             'test': 'Kwiatkowski-Phillips-Schmidt-Shin'
#         }
        
#         return results
    
#     def run_auto_arima_model(data, seasonal, m, max_p, max_d, max_q, ic, stepwise, trace):
#         """Run Auto-ARIMA with optimized parameters"""
        
#         try:
#             # Check stationarity first
#             stationarity_results = check_stationarity(data['Close'])
            
#             # Auto-ARIMA configuration
#             model = auto_arima(
#                 data['Close'],
#                 start_p=0,
#                 start_q=0,
#                 max_p=max_p,
#                 max_d=max_d,
#                 max_q=max_q,
#                 seasonal=seasonal,
#                 m=m,
#                 start_P=0,
#                 start_Q=0,
#                 max_P=max_p,
#                 max_Q=max_q,
#                 information_criterion=ic,
#                 stepwise=stepwise,
#                 trace=trace,
#                 error_action='ignore',
#                 suppress_warnings=True,
#                 n_fits=50,
#                 random_state=42
#             )
            
#             return model, stationarity_results
            
#         except Exception as e:
#             st.error(f"Auto-ARIMA error: {e}")
#             return None, None
    
#     # ==================== LSTM/GRU FUNCTIONS ====================
    
#     def create_sequences(data, lookback=60):
#         X, y = [], []
#         for i in range(lookback, len(data)):
#             X.append(data[i-lookback:i, 0])
#             y.append(data[i, 0])
#         return np.array(X), np.array(y)
    
#     def build_lstm_model(units, lookback):
#         model = Sequential([
#             LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
#             Dropout(0.2),
#             LSTM(units//2, return_sequences=False),
#             Dropout(0.2),
#             Dense(25, activation='relu'),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         return model
    
#     def build_gru_model(units, lookback):
#         model = Sequential([
#             GRU(units, return_sequences=True, input_shape=(lookback, 1)),
#             Dropout(0.2),
#             GRU(units//2, return_sequences=False),
#             Dropout(0.2),
#             Dense(25, activation='relu'),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         return model
    
#     # ==================== TRAINING FUNCTION ====================
    
#     def train_models(data, lookback, test_size, epochs, units, auto_arima_params):
        
#         close_prices = data[['Close']].values
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(close_prices)
        
#         train_size = int(len(scaled_data) * (1 - test_size/100))
        
#         # Train/test split
#         train_prices = close_prices[:train_size].flatten()
#         test_prices = close_prices[train_size:].flatten()
        
#         # ========== AUTO-ARIMA ==========
#         with st.spinner("Running Auto-ARIMA optimization..."):
#             arima_model, stationarity_results = run_auto_arima_model(
#                 data.iloc[:train_size],
#                 auto_arima_params['seasonal'],
#                 auto_arima_params['m'],
#                 auto_arima_params['max_p'],
#                 auto_arima_params['max_d'],
#                 auto_arima_params['max_q'],
#                 auto_arima_params['ic'],
#                 auto_arima_params['stepwise'],
#                 auto_arima_params['trace']
#             )
            
#             if arima_model is not None:
#                 # Get optimal parameters
#                 optimal_params = arima_model.order
#                 seasonal_params = arima_model.seasonal_order
                
#                 # Make predictions
#                 arima_pred = arima_model.predict(n_periods=len(test_prices))
                
#                 # Ensure predictions match test length
#                 if len(arima_pred) < len(test_prices):
#                     last_val = arima_pred[-1] if len(arima_pred) > 0 else train_prices[-1]
#                     arima_pred = np.concatenate([arima_pred, [last_val] * (len(test_prices) - len(arima_pred))])
#                 elif len(arima_pred) > len(test_prices):
#                     arima_pred = arima_pred[:len(test_prices)]
                
#                 arima_pred = np.array(arima_pred).flatten()
#             else:
#                 # Fallback to simple ARIMA
#                 from statsmodels.tsa.arima.model import ARIMA
#                 fallback_model = ARIMA(train_prices, order=(2, 1, 2))
#                 fallback_fitted = fallback_model.fit()
#                 arima_pred = fallback_fitted.forecast(steps=len(test_prices))
#                 optimal_params = (2, 1, 2)
#                 seasonal_params = (0, 0, 0, 0)
#                 stationarity_results = None
        
#         # ========== LSTM ==========
#         with st.spinner("Training LSTM model..."):
#             train_data = scaled_data[:train_size]
#             test_data = scaled_data[train_size - lookback:]
            
#             X_train, y_train = create_sequences(train_data, lookback)
#             X_test, y_test = create_sequences(test_data, lookback)
            
#             X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#             X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
#             lstm_model = build_lstm_model(units, lookback)
#             early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
#             lstm_model.fit(
#                 X_train, y_train,
#                 epochs=epochs,
#                 batch_size=32,
#                 validation_split=0.1,
#                 callbacks=[early_stop],
#                 verbose=0
#             )
            
#             lstm_pred = lstm_model.predict(X_test, verbose=0)
#             lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
        
#         # ========== GRU ==========
#         with st.spinner("Training GRU model..."):
#             gru_model = build_gru_model(units, lookback)
#             gru_model.fit(
#                 X_train, y_train,
#                 epochs=epochs,
#                 batch_size=32,
#                 validation_split=0.1,
#                 callbacks=[early_stop],
#                 verbose=0
#             )
            
#             gru_pred = gru_model.predict(X_test, verbose=0)
#             gru_pred = scaler.inverse_transform(gru_pred).flatten()
        
#         # Ensure all predictions align
#         min_len = min(len(test_prices), len(lstm_pred), len(gru_pred), len(arima_pred))
#         actual_values = test_prices[:min_len]
#         lstm_pred = lstm_pred[:min_len]
#         gru_pred = gru_pred[:min_len]
#         arima_pred = arima_pred[:min_len]
        
#         return {
#             'lstm': lstm_pred,
#             'gru': gru_pred,
#             'arima': arima_pred,
#             'actual': actual_values,
#             'scaler': scaler,
#             'train_size': train_size,
#             'data': data,
#             'last_sequence': scaled_data[-lookback:],
#             'arima_model': arima_model,
#             'optimal_params': optimal_params,
#             'seasonal_params': seasonal_params,
#             'stationarity_results': stationarity_results
#         }
    
#     # ==================== FUTURE PREDICTIONS ====================
    
#     def predict_future(model, last_sequence, scaler, days=30):
#         predictions = []
#         current_seq = last_sequence.copy()
        
#         for _ in range(days):
#             try:
#                 next_pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
#                 predictions.append(next_pred[0, 0])
#                 current_seq = np.roll(current_seq, -1)
#                 current_seq[-1] = next_pred[0, 0]
#             except:
#                 predictions.append(current_seq[-1])
#                 current_seq = np.roll(current_seq, -1)
#                 current_seq[-1] = current_seq[-1]
        
#         try:
#             return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
#         except:
#             return np.array(predictions)
    
#     # ==================== MAIN EXECUTION ====================
    
#     if st.button("🚀 Run Predictions", type="primary"):
        
#         # Fetch data
#         with st.spinner("Fetching stock data..."):
#             data = fetch_data(ticker, start_date, end_date)
        
#         if data.empty:
#             st.error("❌ No data available. Please try different dates.")
#             st.stop()
        
#         # Data info
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("📊 Total Days", len(data))
#         with col2:
#             st.metric("📈 Current Price", f"${data['Close'].iloc[-1]:.2f}")
#         with col3:
#             st.metric("📉 Min Price", f"${data['Close'].min():.2f}")
#         with col4:
#             st.metric("📈 Max Price", f"${data['Close'].max():.2f}")
        
#         # Auto-ARIMA parameters
#         auto_arima_params = {
#             'seasonal': seasonal,
#             'm': m,
#             'max_p': max_p,
#             'max_d': max_d,
#             'max_q': max_q,
#             'ic': information_criterion,
#             'stepwise': stepwise,
#             'trace': trace
#         }
        
#         # Train models
#         results = train_models(
#             data, lookback_days, test_size, epochs, lstm_units,
#             auto_arima_params
#         )
        
#         # ========== STATIONARITY RESULTS ==========
#         if results['stationarity_results']:
#             st.markdown("## 📊 Stationarity Test Results")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("**ADF Test**")
#                 adf_results = results['stationarity_results']['ADF']
#                 st.write(f"p-value: {adf_results['p_value']:.4f}")
#                 st.write(f"Stationary: {'✅' if adf_results['is_stationary'] else '❌'}")
            
#             with col2:
#                 st.markdown("**KPSS Test**")
#                 kpss_results = results['stationarity_results']['KPSS']
#                 st.write(f"p-value: {kpss_results['p_value']:.4f}")
#                 st.write(f"Stationary: {'✅' if kpss_results['is_stationary'] else '❌'}")
        
#         # ========== OPTIMAL PARAMETERS ==========
#         st.markdown("## 🎯 Optimal ARIMA Parameters")
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric("p (AR Order)", results['optimal_params'][0])
#         with col2:
#             st.metric("d (Differencing)", results['optimal_params'][1])
#         with col3:
#             st.metric("q (MA Order)", results['optimal_params'][2])
        
#         if results['seasonal_params'] and any(results['seasonal_params']):
#             st.info(f"📅 Seasonal Order: P={results['seasonal_params'][0]}, D={results['seasonal_params'][1]}, Q={results['seasonal_params'][2]}, m={results['seasonal_params'][3]}")
        
#         # ========== METRICS ==========
#         metrics = {}
#         for model_name in ['lstm', 'gru', 'arima']:
#             pred = results[model_name]
#             actual = results['actual'][-len(pred):]
            
#             pred = np.nan_to_num(pred)
#             actual = np.nan_to_num(actual)
            
#             if len(pred) > 0 and len(actual) > 0:
#                 mse = mean_squared_error(actual, pred)
#                 rmse = np.sqrt(mse)
#                 mae = mean_absolute_error(actual, pred)
#                 r2 = r2_score(actual, pred) if len(actual) > 1 else 0
                
#                 metrics[model_name] = {
#                     'MSE': mse,
#                     'RMSE': rmse,
#                     'MAE': mae,
#                     'R²': max(0, r2)
#                 }
        
#         # Display metrics
#         st.markdown("## 📊 Model Performance Comparison")
#         metrics_df = pd.DataFrame(metrics).T
#         st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', subset=['R²']), use_container_width=True)
        
#         if metrics:
#             best_model = min(metrics, key=lambda x: metrics[x]['RMSE'])
#             st.success(f"🏆 Best Performing Model: **{best_model.upper()}** (RMSE: {metrics[best_model]['RMSE']:.4f})")
        
#         # ========== PREDICTION PLOT ==========
#         st.markdown("## 📈 Model Predictions vs Actual Values")
        
#         fig = go.Figure()
        
#         # Actual
#         actual_dates = data.index[results['train_size']:results['train_size'] + len(results['actual'])]
#         fig.add_trace(go.Scatter(
#             x=actual_dates,
#             y=results['actual'],
#             mode='lines',
#             name='Actual',
#             line=dict(color='black', width=3)
#         ))
        
#         # Predictions
#         colors = {'lstm': 'blue', 'gru': 'green', 'arima': 'red'}
#         for model_name in ['lstm', 'gru', 'arima']:
#             pred_values = results[model_name]
#             pred_dates = actual_dates[:len(pred_values)]
#             fig.add_trace(go.Scatter(
#                 x=pred_dates,
#                 y=pred_values,
#                 mode='lines',
#                 name=f'{model_name.upper()} Prediction',
#                 line=dict(color=colors[model_name], width=1.5, dash='dash')
#             ))
        
#         fig.update_layout(
#             title=f'{selected_company} - Model Predictions vs Actual',
#             xaxis_title='Date',
#             yaxis_title='Price ($)',
#             height=600,
#             hovermode='x unified'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         # ========== 30-DAY FUTURE ==========
#         st.markdown("## 🔮 30-Day Price Predictions")
        
#         with st.spinner("Generating 30-day predictions..."):
#             try:
#                 # LSTM/GRU future
#                 scaler = MinMaxScaler()
#                 close_prices = data[['Close']].values
#                 scaled_full = scaler.fit_transform(close_prices)
                
#                 X_full, y_full = create_sequences(scaled_full, lookback_days)
#                 X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
                
#                 lstm_model = build_lstm_model(lstm_units, lookback_days)
#                 lstm_model.fit(X_full, y_full, epochs=epochs, batch_size=32, verbose=0)
                
#                 gru_model = build_gru_model(lstm_units, lookback_days)
#                 gru_model.fit(X_full, y_full, epochs=epochs, batch_size=32, verbose=0)
                
#                 last_seq = scaled_full[-lookback_days:]
#                 lstm_future = predict_future(lstm_model, last_seq, scaler, 30)
#                 gru_future = predict_future(gru_model, last_seq, scaler, 30)
                
#                 # ARIMA future using fitted model
#                 if results['arima_model'] is not None:
#                     arima_future = results['arima_model'].predict(n_periods=30)
#                     if len(arima_future) < 30:
#                         last_val = arima_future[-1] if len(arima_future) > 0 else data['Close'].iloc[-1]
#                         arima_future = np.concatenate([arima_future, [last_val] * (30 - len(arima_future))])
#                 else:
#                     arima_future = np.array([data['Close'].iloc[-1]] * 30)
                
#                 # Future dates
#                 last_date = data.index[-1]
#                 future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=30)
                
#             except Exception as e:
#                 st.error(f"Error generating predictions: {e}")
#                 st.stop()
        
#         # Create DataFrame
#         future_df = pd.DataFrame({
#             'Date': future_dates,
#             'LSTM': lstm_future,
#             'GRU': gru_future,
#             'ARIMA': arima_future[:30]
#         })
#         future_df['ARIMA'] = future_df['ARIMA'].fillna(future_df['LSTM'])
#         future_df['Average'] = future_df[['LSTM', 'GRU', 'ARIMA']].mean(axis=1)
        
#         st.dataframe(
#             future_df.style.format({
#                 'LSTM': '${:.2f}',
#                 'GRU': '${:.2f}',
#                 'ARIMA': '${:.2f}',
#                 'Average': '${:.2f}'
#             }),
#             use_container_width=True
#         )
        
#         # ========== FUTURE PLOT ==========
#         st.markdown("### 📈 Future Price Trends")
        
#         fig_future = go.Figure()
        
#         # Historical (last 60 days)
#         hist_data = data.iloc[-60:]
#         fig_future.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['Close'].values,
#             mode='lines',
#             name='Historical',
#             line=dict(color='black', width=2)
#         ))
        
#         # Future predictions
#         colors = {'LSTM': 'blue', 'GRU': 'green', 'ARIMA': 'red', 'Average': 'purple'}
#         for model in ['LSTM', 'GRU', 'ARIMA', 'Average']:
#             fig_future.add_trace(go.Scatter(
#                 x=future_dates,
#                 y=future_df[model],
#                 mode='lines+markers',
#                 name=model,
#                 line=dict(color=colors[model], width=2, dash='dot'),
#                 marker=dict(size=6)
#             ))
        
#         fig_future.update_layout(
#             title=f'{selected_company} - 30-Day Price Forecast (Auto-ARIMA Optimized)',
#             xaxis_title='Date',
#             yaxis_title='Price ($)',
#             height=600,
#             hovermode='x unified'
#         )
#         st.plotly_chart(fig_future, use_container_width=True)
        
#         # ========== SUMMARY ==========
#         st.markdown("### 📋 Prediction Summary")
        
#         col1, col2, col3 = st.columns(3)
#         current_price = data['Close'].iloc[-1]
        
#         with col1:
#             lstm_change = ((lstm_future[-1] - current_price) / current_price * 100)
#             st.metric("📈 LSTM", f"${lstm_future[-1]:.2f}", f"{lstm_change:.2f}%")
#         with col2:
#             gru_change = ((gru_future[-1] - current_price) / current_price * 100)
#             st.metric("📈 GRU", f"${gru_future[-1]:.2f}", f"{gru_change:.2f}%")
#         with col3:
#             arima_change = ((arima_future[-1] - current_price) / current_price * 100)
#             st.metric("📈 ARIMA", f"${arima_future[-1]:.2f}", f"{arima_change:.2f}%")
        
#         # ========== DOWNLOAD ==========
#         csv = future_df.to_csv(index=False)
#         st.download_button(
#             "📥 Download Predictions as CSV",
#             data=csv,
#             file_name=f"{ticker}_auto_arima_predictions.csv",
#             mime="text/csv"
#         )
        
#         # ========== MODEL DETAILS ==========
#         with st.expander("📊 Model Details & Diagnostics"):
#             st.markdown("**Auto-ARIMA Model Summary**")
#             if results['arima_model'] is not None:
#                 st.write(f"Optimal Order: {results['optimal_params']}")
#                 st.write(f"Seasonal Order: {results['seasonal_params']}")
#                 st.write(f"Information Criterion: {information_criterion.upper()}")
#                 st.write(f"Stepwise: {stepwise}")
                
#                 # Model summary
#                 st.text(str(results['arima_model'].summary()))
#             else:
#                 st.warning("⚠️ Auto-ARIMA model could not be fitted. Using fallback ARIMA(2,1,2).")
    
#     else:
#         st.info("👈 Adjust parameters and click 'Run Predictions' to start.")
        
#         # ========== HOW IT WORKS ==========
#         st.markdown("""
#         ### 🎯 How Auto-ARIMA Works
        
#         **1. Stationarity Testing**
#         - ADF Test: Checks if data has unit root
#         - KPSS Test: Checks if data is stationary around a trend
        
#         **2. Parameter Optimization**
#         - Tests combinations of (p, d, q)
#         - Optimizes using AIC/BIC/HQIC
#         - Finds best seasonal parameters if enabled
        
#         **3. Model Selection**
#         - Balances accuracy vs complexity
#         - Uses stepwise search for speed
#         - Returns optimal ARIMA configuration
#         """)
        
#         # Create comparison table
#         st.markdown("### 📊 Auto-ARIMA vs Manual ARIMA")
        
#         comparison_data = {
#             "Feature": ["Parameter Selection", "Stationarity Check", "Seasonality", "Speed", "Accuracy"],
#             "Auto-ARIMA": ["✅ Automatic", "✅ Built-in", "✅ Handles", "⚡ Fast", "🎯 High"],
#             "Manual ARIMA": ["❌ Manual", "❌ Manual", "❌ Complex", "🐢 Slow", "🎯 Variable"]
#         }
#         st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

# if __name__ == "__main__":
#     run_stock_prediction()