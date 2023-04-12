import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set title of the web app
st.title('Stock Price Prediction')

# Function to get stock data
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
def get_data(stock):
    df = yf.download(stock, start, end)
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    return df

# Define the list of companies to download data for
tech_list = ['SHOP', 'COST', 'WMT', 'CP', 'AQN', 'TM']
selected_stock = st.selectbox('Select a stock', tech_list + ['Enter Symbol...'])

if selected_stock == 'Enter Symbol...':
    custom_stock = st.text_input('Enter a stock symbol:')
    if custom_stock == '':
        st.warning('Please enter a stock symbol.')
    else:
        selected_stock = custom_stock.upper()
        try:
            print(yf.Ticker(selected_stock).info)
        except:
            st.warning('Invalid stock symbol.')
            selected_stock = None

if selected_stock is not None:
    # Define start and end dates

    yf.pdr_override()

    # Download data
    try:
        df = yf.download(selected_stock, start, end)
    except ValueError:
        st.write(f"Invalid stock symbol: {selected_stock}")
        st.stop()

    # Select only the 'Close' column
    data = df.filter(['Close']).values

    if data.shape[0] == 0 or selected_stock == 'Enter Symbol...':
        st.write("Error: No data available for this stock or you haven't select or entered any stock ")
        st.stop()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Define the length of the training data
    training_data_len = int(np.ceil(len(data) * .95))

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the training data set into x_train and y_train
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = data[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    # Create dataframe for visualization
    train = df.iloc[:training_data_len, :]
    valid = df.iloc[training_data_len:, :]
    valid.loc[:, 'Predictions'] = predictions

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig2.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual'))
    fig2.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))

    fig2.update_layout(
        title='Predictions for ' + selected_stock,
        xaxis_title='Date',
        yaxis_title='Close Price USD ($)',
        hovermode='x',
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig2.update_yaxes(fixedrange=False)
    st.plotly_chart(fig2)

    # Display RMSE
    st.write(f'Root Mean Squared Error: {rmse}')

# Set app title
st.title("Compare two Stocks")

# Define sidebar options
stocks = ["AAPL", "GOOGL", "MSFT", "AMZN"]
selected_stocks = st.sidebar.multiselect("Select stocks to compare", stocks)

# Set date range
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2023-04-11"))

def get_data(stock):
    df = yf.download(stock, start_date, end_date)
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    return df

# Get stock data for selected stocks
data = []
for stock in selected_stocks:
    stock_data = get_data(stock)
    data.append(stock_data)

# Define prophet parameters
prophet_params = {
    "daily_seasonality": False,
    "weekly_seasonality": False,
    "yearly_seasonality": True,
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.05,
    "interval_width": 0.95,
}

# Create a dataframe to store the predicted values
predictions_df = pd.DataFrame()

# Loop through selected stocks and make predictions
for i, stock_data in enumerate(data):
    m = Prophet(**prophet_params)
    m.fit(stock_data)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    forecast["stock"] = selected_stocks[i]
    predictions_df = pd.concat([predictions_df, forecast])

# Plot the predicted values for all selected stocks
if len(selected_stocks) > 0:
    fig = go.Figure()
    for stock in selected_stocks:
        df = predictions_df[predictions_df["stock"] == stock]
        fig.add_trace(go.Scatter(x=df["ds"], y=df["yhat"], name=stock))
    fig.update_layout(title="Predicted Stock Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)
else:
    st.warning("Please select at least one stock.")