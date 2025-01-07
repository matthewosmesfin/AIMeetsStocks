# Stock Price Prediction Project Documentation

This document contains the goals, methods and issues encountered while working on this project.

## Goal

The goal of this project is to predict Stock Prices

## Methods

The main approach used for prediciting Stock Prices was implementing an LSTM Model. An LSTM (Long Short Term Memory) is a type of Recurrent Neural Network
that uses three gates (input, output, forget) that better stores and handles old memory/data and more recent memory/data. An LSTM is most often used for handling sequential data, so that makes it perfect for predicting stock prices, and its robustness lead us to believe that it could predict something as volatile as stock prices.

### Initial Project Setup -- Demo
We first used code pulled from a GitHub repository, that included its own sources of data, and attempted to strengthen the existing code. The LSTM model was changed (by way of reducing the amount used) in order to reduce overfitting. Then hyperparameter tuning was introduced, as well as new sources of data.

All the code existed in a Jupyter Notebook. We then transitioned to making a Stock Predictor class in a Python file, and incporating all the functionality we already had.

After doing so we launched a user interface using streamlit. The UI allowed you to train, test specific pieces of data (only two Tesla and Goodle csvs were valid), as well as have the option to retune the hyperparameters. It then output the prediction percentage as well as a graph with the predicted and actual prices.

#### Issues and Questions Encountered

Questions we asked ourselves during the process:
- Do we need to retrain the model <i>everytime</i>? What if we test a stock on a model that it hasn't been trained on?
- How can scale this to more stocks?
- The UI is clunky and shows the user more than necessary.

### Secondary Project Setup -- Using API
Much of the questions/issues we had were addressed here. We streamlined the UI, put the model training strictly in the backend, and incorporated an API so we can make predictions for <b>any</b> stock.

#### Issues and Questions Encountered

Questions asked:
- How can we also make <b>future</b> predictions
- Is there a way to store API calls so we don't have to request from the URL <i>everytime</i> when making repeated predictions for a single stock?

Despite the fact the the app was much cleaner, and more efficient, there was a major issue discovered when looking into the results of a stock predicton. When looking at the stock predictions, we saw that it was essentially just the actual prices but shifted to the right. After doing some reading I realized this was because the model was not finding any discernable patterns, and so its best option was to use yesterdays price as todays predicted price. This made me question if it was even possible to find any patterns in stock prices that a model can predict, as well as expand the hyperparameters we tuned over. However, the biggest revelation this gave me was that I required significantly more reading into ML and LSTMS in order to understand more how to approach this better.

## Reading/Research
- https://www.semanticscholar.org/reader/29528d8cb030a65f62a35b1237f1f5483077ad0a 
- https://arxiv.org/pdf/2201.11903
