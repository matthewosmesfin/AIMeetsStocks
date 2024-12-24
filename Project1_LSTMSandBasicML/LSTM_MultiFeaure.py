import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Attention
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import keras_tuner
import streamlit as st

#Stock prices in seperate folder

#Function 1: reads data, splits into training and testing if necessary

#Function 2: Hyperparameter tune

#Function 3: Run model, with hyperpamaters tuned

#Function 4: Predict and compare to test data

# Put all of this in a class??
class StockPredictor():
    def __init__(self, StockName):
        self.StockName = StockName
        self.TrainPercent = 0.8
        self.ValidationPercent = 0.2
        self.TrainData = None
        self.TestData = None
        self.X_train = None
        self.y_train = None
        self.TimeStep = 60
        self.BatchSize = None
        self.Dropout = None
        self.Units = None
        self.history = None
        self.Model = None
        self.Trainscaler = None
        self.X_test = None
        self.y_test = None
        self.accuracy = None
        self.predicted_price = None
        self.Retune = False
        self.InputLayer = None

    def preprocess_google(self):
        print("preprocessing")
        self.TrainData["Close"]=pd.to_numeric(self.TrainData.Close,errors='coerce')
        self.TrainData = self.TrainData.dropna()

    functions = {
        'Google': preprocess_google,
    }
   
    def readData(self, FileNameTrain, FileNameTest):
        #check if FileNames exist

        #if second one is None, we know we split from train

        # # if that is the case, train = 80%, test = 20%

        #return training set, testing set, and scaler

        try:
            trainingData = pd.read_csv(FileNameTrain)    
        except:
            print("File not found or not csv file")
       
        if FileNameTest is None:
            splitValue = int(len(trainingData) * self.TrainPercent)
            trainingData, testingData = trainingData.iloc[:splitValue], trainingData.iloc[splitValue:]
            # print("Training Data")
            # print(trainingData)
            # print("Testing Data")
            # print(testingData)
        else:
            try:
                testingData = pd.read_csv(FileNameTest)    
            except:
                print("File not found or not csv file")
        self.TrainData = trainingData
        self.TestData = testingData
        try:
            print("attempt")
            preprocess_function = self.functions[self.StockName]
        except:
            print("no preprocessor ")
        else:
            preprocess_function(self)

        #setting timestep:

        if (len(self.TestData) < 2 * self.TimeStep):
            print("WARNING: test data does not contain enough instances to generate accurate results")
            self.TimeStep = int(len(self.TestData) * 1/3) #arbitrary value for now
   
    def dataPreprocess():
        pass
       
    def processTrain(self, trainData, numCol = 4, predictionCol = 3):
        rows, columns = trainData.shape
        scaler = MinMaxScaler(feature_range=(0,1))

        self.Trainscaler = scaler
       
        for i in range(columns):
            trainData[:,i:i+1] = scaler.fit_transform(trainData[:,i:i+1])

        X_train = []
        y_train = []
       
        for i in range (self.TimeStep, len(trainData)): #
            X_train.append(trainData[i-self.TimeStep:i])
            y_train.append(trainData[i,predictionCol:]) #close happens to be the 4th column in that specific

        X_train,y_train = np.array(X_train),np.array(y_train)

        # print(X_train.shape)
        # print(y_train.shape)

        X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],numCol))
       
        # self.X_train = X_train
        # self.Y_train = y_train
        return X_train, y_train
       
   
    def build_model(self, hp):
        model = Sequential()
        # Tune the number of LSTM units
        units = hp.Int('units', min_value=50, max_value=300, step=50)
        model.add(LSTM(units, input_shape= self.InputLayer, return_sequences=False))
        batch_sizes = hp.Choice('batch_size', [16, 32, 64, 128, 264])
        # Tune the dropout rate
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.6, step=0.1)
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.InputLayer[1]))
        model.compile(
            optimizer='adam', #should we tune this as well?
            loss='mean_squared_error', #how do we decide between them?
            metrics=['accuracy']
        )
        return model

    def hyperParameterTuning(self, X_train, y_train, features, numTrials = 20, executionPerTrial = 1, trainEpochs = 10):
        #use train set to tune hyperparameters

        #split some of train set as validation set

        #return the hyperparameters
        split_ratio = 1 - self.ValidationPercent

        train_size = int(len(X_train) * split_ratio)

        X_train, X_val = X_train[:train_size], X_train[train_size:]
        y_train, y_val = y_train[:train_size], y_train[train_size:]

        print(X_val.shape)
        print(y_val.shape)
        self.InputLayer = X_train.shape()

        X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1], features)) #edit later probably


        tuner = keras_tuner.RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=numTrials,  # Number of different hyperparameter combinations to try
            executions_per_trial=executionPerTrial,  # Number of models to train per combination
            #directory='my_dir', #Where it saves all the trials
            #project_name='lstm_tuning'
        )

        # Perform the search
        tuner.search(
            X_train,
            y_train,
            epochs=trainEpochs, #use less epochs for training to make it faster
            validation_data=(X_val, y_val))

        # Retrieve the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best number of units: {best_hps.get('units')}")
        print(f"Best dropout rate: {best_hps.get('dropout_rate')}")
        print(f"Best batch size: {best_hps.get('batch_size')}")

        self.BatchSize = best_hps.get('batch_size')
        self.Units = best_hps.get('units')
        self.Dropout = best_hps.get('dropout_rate')


    def defineAndTrainModel(self, X_train, y_train, epochs = 60, inputs = 1): #hyperparameters as inputs
        #define our model with the hyperparameters

        #train it

        #return model and history (also output it here?)
        model = Sequential()
        model.add(LSTM(units=self.Units, return_sequences=False, input_shape=(X_train.shape[1], inputs)))
        model.add(Dropout(self.Dropout))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=self.BatchSize, verbose=2)

        self.history = history
        self.Model = model

    def processTest(self, testData, predictionCol = 3, UseNewScaler = False): #assuming preprocesed testSet
        #predict using our training set

        #output prediction with accuracy

        y_test = testData.iloc[self.TimeStep:, predictionCol:].values  # assuming close is the

        # Input array for the model
        inputMulti = testData.values  # Full dataset as NumPy array

        if (UseNewScaler):
            self.Testscaler = MinMaxScaler(feature_range=(0,1))
        else:
            self.Testscaler = self.Trainscaler
        for i in range(testData.shape[1]):
            print(inputMulti[:,i:i+1])
            inputMulti[:,i:i+1] = self.Testscaler.fit_transform(inputMulti[:,i:i+1])  # Scale the data
        inputMulti.shape  # Shape: (length, 4)

        # Prepare the X_test array
        X_test = []
        print("Len of testData: ", len(testData))
        print("Len of inputmult: ", len(testData))
        length = len(testData)

        for i in range(self.TimeStep, length):
            # Append the last 60 timesteps of all features for each row
            X_test.append(inputMulti[i-self.TimeStep:i, :])  # Shape: (60, 4)

        # Convert to NumPy array and reshape
        X_test = np.array(X_test)  # Convert to a NumPy array
        print("X_test shape:", X_test.shape)  # Expected shape: (192, 60, 4)
        print("y_test_multi shape:", y_test.shape)  # Expected shape: (192, 4)

        # self.X_test = X_test
        # self.y_test = y_test
        return X_test, y_test

    def calculate_accuracy(self):
        real = np.array(self.y_test) + 1
        predict = np.array(self.predicted_price) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    def predict(self, X_test, y_test):
        y_pred = self.Model.predict(X_test)
        predicted_price = self.Trainscaler.inverse_transform(y_pred)
        y_test = self.Testscaler.inverse_transform(y_test)

        self.y_test = y_test
        self.predicted_price = predicted_price

        # plt.plot(y_test, color='red', label=f'Actual {self.StockName} Stock Price')
        # plt.plot(predicted_price, color='green', label=f'Predicted {self.StockName} Stock Price')
        # plt.title(f'{self.StockName} Stock Price Prediction')
        # plt.xlabel('Time')
        # plt.ylabel('Stock price')
        # plt.legend()
        # plt.show()

        for i in range(len(y_test)):
            print(f"Y Test: {y_test[i]}, predicted: {predicted_price[i]}")

        self.accuracy = self.calculate_accuracy()

def main():
    stock = StockPredictor("Google")
    stock.readData("./Google_train_data.csv", "./Google_test_data.csv")
   
    X_train, y_train = stock.processTrain(stock.TrainData.iloc[:,1:5].values)
    print(X_train)
    if (stock.Retune):
        stock.hyperParameterTuning(X_train, y_train, 4)
    print(stock.BatchSize, stock.Dropout, stock.Units)
    stock.BatchSize = 16
    stock.Dropout = 0.3
    stock.Units = 50
    stock.defineAndTrainModel(X_train, y_train, inputs=4, epochs=15)
    print(stock.TestData.iloc[:,1:5])
    X_test, y_test = stock.processTest(stock.TestData.iloc[:,1:5])
    print(X_test)
    print(y_test)
    stock.predict(X_test, y_test)
    print(stock.accuracy)
    plt.plot(stock.y_test, color='red', label=f'Actual {stock.StockName} Stock Price')
    plt.plot(stock.predicted_price, color='green', label=f'Predicted {stock.StockName} Stock Price')
    plt.title(f'{stock.StockName} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.legend()
    plt.show()


main()
