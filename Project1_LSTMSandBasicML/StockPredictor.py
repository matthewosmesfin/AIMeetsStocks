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

            stockdf = pd.DataFrame({
                "Open": [float(x) for x in Open],
                "High": [float(x) for x in High],
                "Low": [float(x) for x in Low],
                "Close": [float(x) for x in Close]
            }, index=pd.to_datetime(Dates))

            self.TestData = stockdf
        except:
            print("Error trying to run")

        print(self.TestData.dtypes)

    # def readData(self, FileNameTrain, FileNameTest): #import data
    #     lengthOfFile = 0
    #     trainingData = None
    #     try:
    #         self.TrainData = pd.read_csv(FileNameTrain)
    #         lengthOfFile = len(self.TrainData)
    #     except FileNotFoundError:
    #         raise FileNotFoundError("Training file not found or not a valid CSV file.")
    #     except Exception as e:
    #         raise ValueError(f"Error reading training file: {e}")

       
    #     if FileNameTest is None:
    #         splitValue = int(lengthOfFile * self.TrainPercent)
    #         self.TrainData, self.TestData = self.TrainData.iloc[:splitValue], self.TrainData.iloc[splitValue:]
    #         # print("Training Data")
    #         # print(trainingData)
    #         # print("Testing Data")
    #         # print(testingData)
    #     else:
    #         try:
    #             self.TestData = pd.read_csv(FileNameTest)    
    #         except FileNotFoundError:
    #             raise FileNotFoundError("Testing file not found or not a valid CSV file.")
    #         except Exception as e:
    #             raise ValueError(f"Error reading training file: {e}")

    #     try:
    #         preprocess_function = self.functions[self.StockName]
    #     except:
    #         print("no preprocessor ")
    #     else:
    #         self.TrainData = preprocess_function(self, self.TrainData)
    #         self.TestData = preprocess_function(self, self.TestData)

    #     #setting timestep:

    #     if (len(self.TestData) < 2 * self.TimeStep):
    #         print("WARNING: test data does not contain enough instances to generate accurate results")
    #         self.TimeStep = int(len(self.TestData) * 1/3) #arbitrary value for now
       
    # def processTrain(self, trainData, numCol = 4, predictionCol = 3):
    #     rows, columns = trainData.shape
    #     scaler = MinMaxScaler(feature_range=(0,1))

    #     self.Trainscaler = scaler
       
    #     for i in range(columns):
    #         trainData[:,i:i+1] = scaler.fit_transform(trainData[:,i:i+1])

    #     X_train = []
    #     y_train = []
       
    #     for i in range (self.TimeStep, len(trainData)): #
    #         X_train.append(trainData[i-self.TimeStep:i])
    #         y_train.append(trainData[i,predictionCol:]) #close happens to be the 4th column in that specific

    #     X_train,y_train = np.array(X_train),np.array(y_train)

    #     # print(X_train.shape)
    #     # print(y_train.shape)

    #     X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],numCol))

    #     self.InputLayer = X_train.shape
       
    #     # self.X_train = X_train
    #     # self.Y_train = y_train
    #     return X_train, y_train
       
   
    # def build_model(self, hp):
    #     model = Sequential()
    #     # Tune the number of LSTM units
    #     units = hp.Int('units', min_value=50, max_value=300, step=50)
    #     model.add(LSTM(units, input_shape=(self.InputLayer[1], self.InputLayer[2]), return_sequences=False))
    #     batch_sizes = hp.Choice('batch_size', [16, 32, 64, 128, 264])
    #     # Tune the dropout rate
    #     dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.6, step=0.1)
    #     model.add(Dropout(dropout_rate))
    #     model.add(Dense(self.InputLayer[1]))
    #     model.compile(
    #         optimizer='adam', #should we tune this as well?
    #         loss='mean_squared_error', #how do we decide between them?
    #         metrics=['mean_absolute_error']
    #     )
    #     return model

    # def hyperParameterTuning(self, X_train, y_train, features, numTrials = 50, executionPerTrial = 1, trainEpochs = 10):
    #     #use train set to tune hyperparameters

    #     #split some of train set as validation set

    #     #return the hyperparameters
    #     split_ratio = 1 - self.ValidationPercent

    #     train_size = int(len(X_train) * split_ratio)

    #     X_train, X_val = X_train[:train_size], X_train[train_size:]
    #     y_train, y_val = y_train[:train_size], y_train[train_size:]

    #     # print(X_val.shape)
    #     # print(y_val.shape)

    #     X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1], features)) #edit later probably


    #     tuner = keras_tuner.RandomSearch(
    #         self.build_model,
    #         objective='val_mean_absolute_error',
    #         max_trials=numTrials,  # Number of different hyperparameter combinations to try
    #         executions_per_trial=executionPerTrial,  # Number of models to train per combination
    #         #directory='my_dir', #Where it saves all the trials
    #         #project_name='lstm_tuning'
    #     )

    #     # Perform the search
    #     tuner.search(
    #         X_train,
    #         y_train,
    #         epochs=trainEpochs, #use less epochs for training to make it faster
    #         validation_data=(X_val, y_val))

    #     # Retrieve the best hyperparameters
    #     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #     print(f"Best number of units: {best_hps.get('units')}")
    #     print(f"Best dropout rate: {best_hps.get('dropout_rate')}")
    #     print(f"Best batch size: {best_hps.get('batch_size')}")

    #     self.BatchSize = best_hps.get('batch_size')
    #     self.Units = best_hps.get('units')
    #     self.Dropout = best_hps.get('dropout_rate')


    # def defineAndTrainModel(self, X_train, y_train, epochs = 60, inputs = 1): #hyperparameters as inputs
    #     #define our model with the hyperparameters

    #     #train it

    #     #return model and history (also output it here?)
    #     model = Sequential()
    #     model.add(LSTM(units=self.Units, return_sequences=False, input_shape=(X_train.shape[1], inputs)))
    #     model.add(Dropout(self.Dropout))
    #     model.add(Dense(self.InputLayer[1]))
    #     model.compile(optimizer='adam', loss='mean_squared_error')

    #     # Train the model
    #     history = model.fit(X_train, y_train, epochs=epochs, batch_size=self.BatchSize, verbose=2)

    #     self.history = history
    #     self.Model = model

    def processTest(self, predictionCol = 3): #assuming preprocesed testSet
        #predict using our training set

        #output prediction with accuracy

        testData = self.TestData.iloc[:,]
        print(testData)
        y_test = testData.iloc[self.TimeStep:, predictionCol:].values  # assuming close is the

        # Input array for the mode
        inputMulti = testData.values  # Full dataset as NumPy array

        self.Scaler = MinMaxScaler(feature_range=(0,1))
        for i in range(testData.shape[1]):
            inputMulti[:,i:i+1] = self.Scaler.fit_transform(inputMulti[:,i:i+1])  # Scale the data
        inputMulti.shape  # Shape: (length, 4)

        # Prepare the X_test array
        X_test = []
        print("Len of testData: ", len(self.TestData))
        print("Len of inputmult: ", len(self.TestData))
        length = len(self.TestData)

        for i in range(self.TimeStep, length):
            # Append the last 60 timesteps of all features for each row
            X_test.append(inputMulti[i-self.TimeStep:i, :])  # Shape: (60, 4)

        # Convert to NumPy array and reshape
        X_test = np.array(X_test)  # Convert to a NumPy array
        y_test = np.array(y_test)
        print("X_test shape:", X_test.shape)  # Expected shape: (80, 20, 4)
        print("y_test_multi shape:", y_test.shape)  # Expected shape: (80, 1)

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

        self.y_test = y_test
        self.predicted_price = predicted_price

        self.accuracy = self.calculate_accuracy()

    def run(self):

        # if train_file is not None:
        #     self.readData(train_file, test_file)
        #     X_train, y_train = self.processTrain(self.TrainData.iloc[:,1:5].values)
        #     if not self.SkipTraining or self.Model is None:
        #         if (self.Retune):
        #             self.hyperParameterTuning(X_train, y_train, 4)
        #         self.defineAndTrainModel(X_train, y_train, inputs=4, epochs=15)
        # # elif test_file is not None:
        # #     self.processTest(test_file)

        self.readData()
        self.setStockName()
        X_test, y_test = self.processTest()
        self.plot_method(X_test, y_test)
        return X_test, y_test
    
    def plot_method(self, X_test, y_test):

        self.predict(X_test, y_test)
        print(self.accuracy)

        plt.plot(self.y_test, color='red', label=f'Actual {self.StockName} self Price')
        plt.plot(self.predicted_price, color='green', label=f'Predicted {self.StockName} self Price')
        plt.title(f'{self.StockName} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock price')
        plt.legend()

        # plt.show()

