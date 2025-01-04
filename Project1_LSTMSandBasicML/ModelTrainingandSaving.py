#Responsbile to training and saving the model we want (as well as tuning hyperparameters)
#Much easier if we put that functionality in this file, instead of it all being in the Stock Predictor file
#We are using a much larger data set to train

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Attention
import tensorflow as tf
import keras_tuner
import joblib

#Import data and read it
data = pd.read_csv('Google_train_data.csv')
data["Close"]=pd.to_numeric(data.Close,errors='coerce')
data = data.dropna()

trainData = data.iloc[:,1:5].values
rows, columns = trainData.shape
scMulti = MinMaxScaler(feature_range=(0,1))

for i in range(columns):
    trainData[:,i:i+1] = scMulti.fit_transform(trainData[:,i:i+1])

X_train = []
y_train = []

for i in range (60,1149): #60 : timestep // 1149 : length of the data
    X_train.append(trainData[i-60:i]) 
    y_train.append(trainData[i,3:])

X_train, y_train = np.array(X_train),np.array(y_train)
lenofData = len(X_train)

# Split data into training and validation sets (80% train, 20% validation)
split_ratio = 0.8  # 80% for training, 20% for validation
train_size = int(lenofData * split_ratio)


# Create the training set (80%) and validation set (20%)
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]
X_val.shape

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],4)) #adding the batch_size axis
X_train.shape

def build_model(hp):
    model = Sequential()
    # Tune the number of LSTM units
    units = hp.Int('units', min_value=50, max_value=300, step=50)
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    batch_sizes = hp.Choice('batch_size', [16, 32, 64, 128, 264])
    # Tune the dropout rate
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.6, step=0.1)
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(
        optimizer='adam', #should we tune this as well?
        loss='mean_squared_error', #how do we decide between them?
        metrics=['mean_absolute_error']
    )
    return model

# Initialize the tuner
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=20,  # Number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to train per combination
    #directory='my_dir', #Where it saves all the trials
    #project_name='lstm_tuning'
)

# Perform the search
tuner.search(
    X_train,
    y_train,
    epochs=10, #use less epochs for training to make it faster
    validation_data=(X_val, y_val))

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of units: {best_hps.get('units')}")
print(f"Best dropout rate: {best_hps.get('dropout_rate')}")
print(f"Best batch size: {best_hps.get('batch_size')}")

units = best_hps.get('units')
dropout = best_hps.get('dropout_rate')
batch_size = best_hps.get('batch_size')

def defineAndTrainModel(X_train, y_train, epochs = 60, inputs = 4): #hyperparameters as inputs
    #define our model with the hyperparameters

    #train it

    #return model and history (also output it here?)
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(X_train.shape[1], inputs)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs = 60, batch_size=batch_size, verbose=2)

    return model

model = defineAndTrainModel(X_train, y_train)
joblib.dump(model, "model.joblib")

