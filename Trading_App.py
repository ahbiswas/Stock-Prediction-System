import streamlit as st

st.title("📈 Trading Intelligence Platform")
st.markdown("#### Empowering Investors with Data-Driven Insights - A H Biswas")
st.set_page_config(page_title="Stock Analysis",
                    page_icon=":page_with_curl:",
                    layout="wide")

st.markdown("""
### Advanced Stock Analysis & Price Forecasting

Welcome to the Trading Intelligence Platform, a data-driven solution designed to help investors and traders make informed decisions in the financial markets.

Our platform provides comprehensive stock analysis, including historical price trends, technical indicators, company fundamentals, and interactive visualizations. Users can explore market behavior through candlestick charts, moving averages, RSI, MACD, and other technical analysis tools.

In addition, the platform offers AI-powered stock price forecasting based on historical market data and time-series modeling techniques. These predictive insights help users identify potential market trends and evaluate future price movements.

Whether you are a beginner investor or an experienced trader, our tools are designed to transform complex financial data into actionable insights.
""")

st.video("create_a_animation_vedio_of_a.mp4")

st.markdown("---")

st.markdown("##  Core Services")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Stock Analysis
    
    - Historical Stock Performance
    - Company Fundamentals
    - Technical Indicators
    - Candlestick Visualization
    - RSI Analysis
    - MACD Analysis
    - Moving Average Trends
    - Market Statistics
    """)

with col2:
    st.markdown("""
    ### 🤖 Stock Price Prediction
    
    - AI-Based Forecasting
    - Time Series Analysis
    - Trend Prediction
    - Future Price Estimation
    - Interactive Forecast Visualizations
    - Data-Driven Investment Insights
    - Short-Term & Long-Term Forecasting
    """)

st.markdown("---")

st.success(
    "Use the navigation menu to access Stock Analysis and Stock Prediction modules."
)

