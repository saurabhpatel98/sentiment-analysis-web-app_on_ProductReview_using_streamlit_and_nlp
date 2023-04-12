!pip install keras
!pip install keras tensorflow
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime
import matplotlib.pyplot as plt

# Define the list of companies to download data for
tech_list = ['SHOP', 'COST', 'WMT', 'CP', 'AQN', 'TM']

# Define start and end dates
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

# Download data for each company and perform prediction
for stock in tech_list:
    # Download data
    df = pdr.get_data_yahoo(stock, start=start, end=end)

    # Select only the 'Close' column
    data = df.filter(['Close']).values

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
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16, 6))
    plt.title(f'{stock} Stock Price Prediction')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')