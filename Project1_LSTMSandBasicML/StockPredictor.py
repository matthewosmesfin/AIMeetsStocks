import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Attention
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
# import keras_tuner
import requests
import yfinance as yf
import joblib

class StockPredictor():
    def __init__(self, StockSymbol):
        self.StockSymbol = StockSymbol #make it stock symbol instead of name?
        self.StockName = None
        self.TrainData = pd.DataFrame()
        self.TestData = pd.DataFrame()
        self.X_train = None
        self.y_train = None
        self.TimeStep = 20
        self.history = None
        self.Model = None
        self.Scaler = None
        self.X_test = None
        self.y_test = None
        self.accuracy = None
        self.predicted_price = None
        self.dates = None

    #process data into X_test and y_test
    #use it to predict
    #that is it

    def setStockName(self):
        try:
            stock = yf.Ticker(self.StockSymbol)
            self.StockName = stock.info.get('shortName', 'Unknown')
        except Exception as e:
            print('Error fetching name')

    def readData(self):
        with open('/Users/matthewosmesfin/Documents/apikey', 'r') as file:
            KEY = file.read()
        
        SYMBOL = self.StockSymbol

        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={KEY}'

        try:
            r = requests.get(url)
            data = r.json()

            print("Got request")

            o = '1. open'
            h = '2. high'
            l = '3. low'
            c = '4. close'

            Open = []
            High = []
            Low = []
            Close = []
            Dates = []

            for key,value in data["Time Series (Daily)"].items():
                Dates.append(key)
                Open.append(value[o])
                High.append(value[h])
                Low.append(value[l])
                Close.append(value[c])

            print("Here?")
            Dates = Dates[::-1]
            Open = Open[::-1]
            High = High[::-1]
            Low = Low[::-1]
            Close = Close[::-1]

            print("Here??")

            stockdf = pd.DataFrame({
                "Open": [float(x) for x in Open],
                "High": [float(x) for x in High],
                "Low": [float(x) for x in Low],
                "Close": [float(x) for x in Close]
            }, index=pd.to_datetime(Dates))

            print("Here??")
            self.TestData = stockdf
            self.dates = Dates
        except:
            print("Error trying to run")

        stockdf.to_csv('work.csv')

        # print(self.TestData.dtypes)

    def processTest(self, predictionCol = 3): #assuming preprocesed testSet
        #predict using our training set

        #output prediction with accuracy

        testData = self.TestData.iloc[:,]
        # testData = self.TestData.iloc[::-1, :]
        # print(testData)
        y_test = testData.iloc[self.TimeStep:, predictionCol:].values  # assuming close is the

        # Input array for the mode
        inputMulti = testData.values  # Full dataset as NumPy array

        print(inputMulti)
        print(inputMulti.shape)

        self.Scaler = MinMaxScaler(feature_range=(0,1))
        for i in range(testData.shape[1]):
            inputMulti[:,i:i+1] = self.Scaler.fit_transform(inputMulti[:,i:i+1])  # Scale the data
        inputMulti.shape  # Shape: (length, 4)

        # Prepare the X_test array
        X_test = []
        length = len(self.TestData)

        for i in range(self.TimeStep, length):
            # Append the last 60 timesteps of all features for each row
            X_test.append(inputMulti[i-self.TimeStep:i, :])  # Shape: (60, 4)

        # Convert to NumPy array and reshape
        X_test = np.array(X_test)  # Convert to a NumPy array
        y_test = np.array(y_test)
        # print("X_test shape:", X_test.shape)  # Expected shape: (80, 20, 4)
        # print("y_test_multi shape:", y_test.shape)  # Expected shape: (80, 1)

        # self.X_test = X_test
        # self.y_test = y_test
        return X_test, y_test

    def calculate_accuracy(self):
        real = np.array(self.y_test) + 1
        predict = np.array(self.predicted_price) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    def predict(self, X_test, y_test):
        model = joblib.load('model.joblib')
        y_pred = model.predict(X_test)
        predicted_price = self.Scaler.inverse_transform(y_pred)
        y_test = self.Scaler.inverse_transform(y_test)

        y_test = pd.Series(y_test.ravel(), index=self.dates[20:])
        predicted_price = pd.Series(predicted_price.ravel(), index=self.dates[20:])

        self.y_test = y_test
        self.predicted_price = predicted_price

        self.accuracy = self.calculate_accuracy()

    def run(self):

        self.readData()
        X_test, y_test = self.processTest()
        self.predict(X_test, y_test)
        # self.plot_method(X_test, y_test)
        return X_test, y_test
    
    def plot_method(self, X_test, y_test):

        self.predict(X_test, y_test)
        # print(self.accuracy)

        plt.plot(self.y_test, color='red', label=f'Actual {self.StockName} self Price')
        plt.plot(self.predicted_price, color='green', label=f'Predicted {self.StockName} self Price')
        plt.title(f'{self.StockName} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock price')
        plt.legend()

        # plt.show()

