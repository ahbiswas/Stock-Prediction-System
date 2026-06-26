import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import ta
from pages.utils.plotly_figure import (
    candelstick,
    close_chart,
    plotly_table,
    RSI,
    Moving_average,
    MACD
)

def run_stock_analysis():
    st.title("Stock Analysis :page_with_curl:")
    st.set_page_config(page_title="Stock Analysis",
                    page_icon=":page_with_curl:",
                    layout="wide")

    col1, col2, col3 = st.columns(3)
    today = dt.date.today()

    with col1:
        companies = {
            "Tesla (TSLA)": "TSLA",
            "Apple (AAPL)": "AAPL",
            "Alphabet Inc. (GOOGL)": "GOOGL",
            "Amazon (AMZN)": "AMZN",
            "Microsoft (MSFT)": "MSFT",
            "Meta Platforms (META)": "META",
            "NVIDIA (NVDA)": "NVDA",
            "Netflix (NFLX)": "NFLX",
            "Adobe (ADBE)": "ADBE",
            "Salesforce (CRM)": "CRM",
            "Intel (INTC)": "INTC",
            "IBM (IBM)": "IBM",
            "Visa (V)": "V",
            "Mastercard (MA)": "MA",
            "PayPal (PYPL)": "PYPL",
            "Shopify (SHOP)": "SHOP",
            "Zoom Video (ZM)": "ZM",
            "Twilio (TWLO)": "TWLO",
            "Uber (UBER)": "UBER",
            "Lyft (LYFT)": "LYFT",
            "Airbnb (ABNB)": "ABNB",
            "Spotify (SPOT)": "SPOT"
        }
        selected_company = st.selectbox("Select Company", list(companies.keys()), index=0)
        ticker = companies[selected_company]
    
    with col2:
        start_date = st.date_input("Choose Start Date", dt.date(today.year-1, today.month, today.day))
    with col3:
        end_date = st.date_input("Choose End Date", dt.date(today.year, today.month, today.day))

    st.subheader(ticker)
    stock = yf.Ticker(ticker)
    
    # Display company info
    st.write(stock.info['longBusinessSummary'])
    st.write(f"**Sector:** {stock.info['sector']}")
    st.write(f"**Industry:** {stock.info['industry']}")
    st.write(f"**Country:** {stock.info['country']}")
    st.write(f"**Market Cap:** {stock.info['marketCap']}")
    st.write(f"**Beta:** {stock.info['beta']}")
    st.write(f"**52 Week High:** {stock.info['fiftyTwoWeekHigh']}")
    st.write(f"**52 Week Low:** {stock.info['fiftyTwoWeekLow']}")
    st.write(f"**Full Time Employees:** {stock.info['fullTimeEmployees']}")
    st.write(f"**Website:** {stock.info['website']}")
    
    col1, col2 = st.columns(2)
    with col1:
        df = pd.DataFrame(index=["Market cap", "Beta", "EPS", "PE Ratio","Quick Ratio", "Revenue per share", "Profit Margins", "Debt to Equity", "Return on Equity"])
        df[''] = [stock.info['marketCap'], stock.info['beta'], stock.info['trailingEps'], stock.info['trailingPE'], stock.info['quickRatio'], stock.info['revenuePerShare'], stock.info['profitMargins'], stock.info['debtToEquity'], stock.info['returnOnEquity']]
        fig = plotly_table(df)
        st.plotly_chart(fig, use_container_width=True)

    data = yf.download(ticker, start=start_date, end=end_date)
    
    col1, col2, col3 = st.columns(3)
    daily_change = (data["Close"].iloc[-1] - data["Close"].iloc[-2]).item()
    col1.metric("Daily Change", str(round(data['Close'].iloc[-1], 2).item()), str(round(daily_change, 2)))

    last_10_df = data.tail(10).sort_index(ascending=False)
    fig = plotly_table(last_10_df)
    st.write("### Last 10 Days Stock Data")
    st.plotly_chart(fig, use_container_width=True)
    
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    num_period = ''
    with col1:
        if st.button("5D"):
            num_period = '5d'
    with col2:
        if st.button("1M"):
            num_period = '1mo'
    with col3:
        if st.button("6M"):
            num_period = '6mo'
    with col4:
        if st.button("1Y"):
            num_period = '1y'
    with col5:
        if st.button("5Y"):
            num_period = '5y'
    with col6:
        if st.button("Max"):
            num_period = 'max'

    col1, col2, col3 = st.columns([1,1,4])
    with col1: 
        chart_type = st.selectbox('Chart Type', ('Line', 'Candles'))
    with col2:
        if chart_type == 'Candles':
            indicators = st.selectbox('Indicator', ('RSI', 'MACD'))
        else:
            indicators = st.selectbox('Indicator', ('RSI', 'Moving Average', 'MACD'))

    ticker_ = yf.Ticker(ticker)
    data1 = ticker_.history(period='max')
    
    if num_period == '':
        if chart_type == 'Candles' and indicators == 'RSI':
            st.plotly_chart(candelstick(data1, '1y'), use_container_width=True)
            st.plotly_chart(RSI(data1, '1y'), use_container_width=True)
        elif chart_type == 'Candles' and indicators == 'MACD':
            st.plotly_chart(candelstick(data1, '1y'), use_container_width=True)
            st.plotly_chart(MACD(data1, '1y'), use_container_width=True)
        elif chart_type == 'Line' and indicators == 'RSI':
            st.plotly_chart(close_chart(data1, '1y'), use_container_width=True)
            st.plotly_chart(RSI(data1, '1y'), use_container_width=True)
        elif chart_type == 'Line' and indicators == 'Moving Average':
            st.plotly_chart(Moving_average(data1, '1y'), use_container_width=True)
        elif chart_type == 'Line' and indicators == 'MACD':
            st.plotly_chart(close_chart(data1, '1y'), use_container_width=True)
            st.plotly_chart(MACD(data1, '1y'), use_container_width=True)
    else:
        if chart_type == 'Candles' and indicators == 'RSI':
            st.plotly_chart(candelstick(data1, num_period), use_container_width=True)
            st.plotly_chart(RSI(data1, num_period), use_container_width=True)
        elif chart_type == 'Candles' and indicators == 'MACD':
            st.plotly_chart(candelstick(data1, num_period), use_container_width=True)
            st.plotly_chart(MACD(data1, num_period), use_container_width=True)
        elif chart_type == 'Line' and indicators == 'RSI':
            st.plotly_chart(close_chart(data1, num_period), use_container_width=True)
            st.plotly_chart(RSI(data1, num_period), use_container_width=True)
        elif chart_type == 'Line' and indicators == 'Moving Average':
            st.plotly_chart(Moving_average(data1, num_period), use_container_width=True)
        elif chart_type == 'Line' and indicators == 'MACD':
            st.plotly_chart(close_chart(data1, num_period), use_container_width=True)
            st.plotly_chart(MACD(data1, num_period), use_container_width=True)

if __name__ == "__main__":
    run_stock_analysis()