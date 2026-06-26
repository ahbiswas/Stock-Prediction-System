# import plotly.graph_objects as go
# import dateutil
# import datetime as dt
# import pandas as pd

# def plotly_table(dataframe):
#     headerColour = 'grey'
#     rowEvenColour = '#f8fafd'
#     rowOddColour = '#e1efff'

#     fig = go.Figure(data=[go.Table(
#         header=dict(
#             values=["<b></b>"] + ["<b>" + (i[0] if isinstance(i, tuple) else str(i)) + "</b>" for i in dataframe.columns],
#             line_color='darkslategray',
#             fill_color=headerColour,
#             align='center',
#             font=dict(color='white', size=15),
#             height=35
#         ),
#         cells=dict(
#             values=[["<b>" + str(i) + "</b>" for i in dataframe.index]] + [dataframe[i] for i in dataframe.columns],line_color='darkslategray',fill_color=[[rowOddColour, rowEvenColour] * len(dataframe)],
#             align='left',font=dict(color='darkslategray', size=12)
#         )
#     )])

#     fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
#     return fig

# def filter_data(dataframe, num_periods):

#     # Handle yfinance MultiIndex columns
#     if isinstance(dataframe.columns, pd.MultiIndex):
#         dataframe.columns = dataframe.columns.get_level_values(0)

#     last_date = dataframe.index[-1]

#     if num_periods == '5d':
#         date = last_date + dateutil.relativedelta.relativedelta(days=-5)

#     elif num_periods == '1mo':
#         date = last_date + dateutil.relativedelta.relativedelta(months=-1)

#     elif num_periods == '3mo':
#         date = last_date + dateutil.relativedelta.relativedelta(months=-3)

#     elif num_periods == '6mo':
#         date = last_date + dateutil.relativedelta.relativedelta(months=-6)

#     elif num_periods == '1y':
#         date = last_date + dateutil.relativedelta.relativedelta(years=-1)

#     elif num_periods == '5y':
#         date = last_date + dateutil.relativedelta.relativedelta(years=-5)

#     else:
#         date = dataframe.index[0]

#     df = dataframe.reset_index()

#     # Rename first column to Date
#     df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

#     # Remove timezone information
#     df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

#     # Convert comparison date to timezone-naive Timestamp
#     date = pd.Timestamp(date).tz_localize(None)

#     return df[df['Date'] > date]

# def close_chart(dataframe, num_period = False):
#     if num_period:
#         dataframe = filter_data(dataframe, num_period)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'], mode='lines', name='Open',line=dict(color='#5ab7ff', width=2)))
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'], mode='lines', name='Close',line=dict(color='#ff7f0e', width=2)))
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'], mode='lines', name='High',line=dict(color='#2ca02c', width=2)))
#     fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'], mode='lines', name='Low',line=dict(color='#d62728', width=2)))
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white',paper_bgcolor="white",legend = dict(yanchor="top", xanchor="left"))
#     return fig

# def candelstick(dataframe, num_period):
#     dataframe =filter_data(dataframe, num_period)
#     fig = go.Figure(data=[go.Candlestick(x=dataframe['Date'],
#                     open=dataframe['Open'], high=dataframe['High'],
#                     low=dataframe['Low'], close=dataframe['Close'])])
#     fig.update_layout(showlegend=False, height=500, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white', paper_bgcolor="white")
#     return fig

# def calculate_rsi(series, period=14):
#     delta = series.diff()

#     gain = delta.where(delta > 0, 0).rolling(period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

#     rs = gain / loss
#     return 100 - (100 / (1 + rs))
# def RSI(dataframe, num_period):

#     if isinstance(dataframe.columns, pd.MultiIndex):
#         dataframe.columns = dataframe.columns.get_level_values(0)

#     dataframe["RSI"] = calculate_rsi(dataframe["Close"])

#     dataframe = filter_data(dataframe, num_period)

#     fig = go.Figure()

#     fig.add_trace(
#         go.Scatter(
#             x=dataframe['Date'],
#             y=dataframe['RSI'],
#             mode='lines',
#             name='RSI'
#         )
#     )

#     fig.update_layout(
#         height=300,
#         yaxis_range=[0, 100],
#         paper_bgcolor="white"
#     )

#     return fig

# def Moving_average(dataframe, num_period):

#     if isinstance(dataframe.columns, pd.MultiIndex):
#         dataframe.columns = dataframe.columns.get_level_values(0)

#     dataframe["MA50"] = dataframe["Close"].rolling(window=50).mean()

#     dataframe = filter_data(dataframe, num_period)

#     fig = go.Figure()

#     # Close Price
#     fig.add_trace(
#         go.Scatter(
#             x=dataframe['Date'],
#             y=dataframe['Close'],
#             mode='lines',
#             name='Close',
#             line=dict(color='#1f77b4', width=2)
#         )
#     )

#     # High Price
#     fig.add_trace(
#         go.Scatter(
#             x=dataframe['Date'],
#             y=dataframe['High'],
#             mode='lines',
#             name='High',
#             line=dict(color='green', width=1)
#         )
#     )

#     # Low Price
#     fig.add_trace(
#         go.Scatter(
#             x=dataframe['Date'],
#             y=dataframe['Low'],
#             mode='lines',
#             name='Low',
#             line=dict(color='red', width=1)
#         )
#     )

#     # MA50
#     fig.add_trace(
#         go.Scatter(
#             x=dataframe['Date'],
#             y=dataframe['MA50'],
#             mode='lines',
#             name='MA50',
#             line=dict(color='orange', width=3)
#         )
#     )

#     fig.update_xaxes(rangeslider_visible=True)

#     fig.update_layout(
#         title='Price with MA50',
#         height=550,
#         margin=dict(l=0, r=0, t=40, b=0),
#         plot_bgcolor='white',
#         paper_bgcolor="white",
#     )

#     return fig

# def MACD(dataframe, num_period):

#     if isinstance(dataframe.columns, pd.MultiIndex):
#         dataframe.columns = dataframe.columns.get_level_values(0)

#     # MACD Calculation
#     ema12 = dataframe["Close"].ewm(span=12, adjust=False).mean()
#     ema26 = dataframe["Close"].ewm(span=26, adjust=False).mean()

#     dataframe["MACD"] = ema12 - ema26
#     dataframe["Signal"] = dataframe["MACD"].ewm(span=9, adjust=False).mean()
#     dataframe["Histogram"] = dataframe["MACD"] - dataframe["Signal"]

#     dataframe = filter_data(dataframe, num_period)

#     fig = go.Figure()

#     # MACD Line
#     fig.add_trace(
#         go.Scatter(
#             x=dataframe["Date"],
#             y=dataframe["MACD"],
#             mode="lines",
#             name="MACD",
#             line=dict(color="#113d5d", width=2)
#         )
#     )

#     # Signal Line
#     fig.add_trace(
#         go.Scatter(
#             x=dataframe["Date"],
#             y=dataframe["Signal"],
#             mode="lines",
#             name="Signal",
#             line=dict(color="#ff7f0e", width=2)
#         )
#     )

#     # Histogram
#     fig.add_trace(
#         go.Bar(
#             x=dataframe["Date"],
#             y=dataframe["Histogram"],
#             name="Histogram"
#         )
#     )

#     fig.update_layout(
#         height=350,
#         margin=dict(l=0, r=0, t=0, b=0),
#         plot_bgcolor="white",
#         paper_bgcolor="white",
#     )

#     return fig

import plotly.graph_objects as go
import dateutil
import datetime as dt
import pandas as pd

def plotly_table(dataframe):
    headerColour = 'grey'
    rowEvenColour = '#f8fafd'
    rowOddColour = '#e1efff'

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b></b>"] + ["<b>" + (i[0] if isinstance(i, tuple) else str(i)) + "</b>" for i in dataframe.columns],
            line_color='darkslategray',
            fill_color=headerColour,
            align='center',
            font=dict(color='white', size=15),
            height=35
        ),
        cells=dict(
            values=[["<b>" + str(i) + "</b>" for i in dataframe.index]] + [dataframe[i] for i in dataframe.columns],
            line_color='darkslategray',
            fill_color=[[rowOddColour, rowEvenColour] * len(dataframe)],
            align='left',
            font=dict(color='darkslategray', size=12)
        )
    )])

    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def filter_data(dataframe, num_periods):
    # Handle yfinance MultiIndex columns
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)

    last_date = dataframe.index[-1]

    if num_periods == '5d':
        date = last_date + dateutil.relativedelta.relativedelta(days=-5)
    elif num_periods == '1mo':
        date = last_date + dateutil.relativedelta.relativedelta(months=-1)
    elif num_periods == '3mo':
        date = last_date + dateutil.relativedelta.relativedelta(months=-3)
    elif num_periods == '6mo':
        date = last_date + dateutil.relativedelta.relativedelta(months=-6)
    elif num_periods == '1y':
        date = last_date + dateutil.relativedelta.relativedelta(years=-1)
    elif num_periods == '5y':
        date = last_date + dateutil.relativedelta.relativedelta(years=-5)
    else:
        date = dataframe.index[0]

    df = dataframe.reset_index()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    date = pd.Timestamp(date).tz_localize(None)

    return df[df['Date'] > date]

def close_chart(dataframe, num_period = False):
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'], mode='lines', name='Open',line=dict(color='#5ab7ff', width=2)))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'], mode='lines', name='Close',line=dict(color='#ff7f0e', width=2)))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'], mode='lines', name='High',line=dict(color='#2ca02c', width=2)))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'], mode='lines', name='Low',line=dict(color='#d62728', width=2)))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white',paper_bgcolor="white",legend = dict(yanchor="top", xanchor="left"))
    return fig

def candelstick(dataframe, num_period):
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure(data=[go.Candlestick(x=dataframe['Date'],
                    open=dataframe['Open'], high=dataframe['High'],
                    low=dataframe['Low'], close=dataframe['Close'])])
    fig.update_layout(showlegend=False, height=500, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white', paper_bgcolor="white")
    return fig

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def RSI(dataframe, num_period):
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)
    dataframe["RSI"] = calculate_rsi(dataframe["Close"])
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['RSI'], mode='lines', name='RSI'))
    fig.update_layout(height=300, yaxis_range=[0, 100], paper_bgcolor="white")
    return fig

def Moving_average(dataframe, num_period):
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)
    dataframe["MA50"] = dataframe["Close"].rolling(window=50).mean()
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'], mode='lines', name='Close', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'], mode='lines', name='High', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'], mode='lines', name='Low', line=dict(color='red', width=1)))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['MA50'], mode='lines', name='MA50', line=dict(color='orange', width=3)))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(title='Price with MA50', height=550, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor='white', paper_bgcolor="white")
    return fig

def MACD(dataframe, num_period):
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)
    ema12 = dataframe["Close"].ewm(span=12, adjust=False).mean()
    ema26 = dataframe["Close"].ewm(span=26, adjust=False).mean()
    dataframe["MACD"] = ema12 - ema26
    dataframe["Signal"] = dataframe["MACD"].ewm(span=9, adjust=False).mean()
    dataframe["Histogram"] = dataframe["MACD"] - dataframe["Signal"]
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["MACD"], mode="lines", name="MACD", line=dict(color="#113d5d", width=2)))
    fig.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["Signal"], mode="lines", name="Signal", line=dict(color="#ff7f0e", width=2)))
    fig.add_trace(go.Bar(x=dataframe["Date"], y=dataframe["Histogram"], name="Histogram"))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white", paper_bgcolor="white")
    return fig