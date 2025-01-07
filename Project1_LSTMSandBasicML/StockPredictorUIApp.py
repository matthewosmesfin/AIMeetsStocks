import streamlit as st
import pandas as pd
from StockPredictor import StockPredictor  # Assuming the class is saved as stock_predictor.py
import yfinance as yf


#WHAT IS LEFT

#add dates to graph
#include next day prediction
#make graph more readable (colors, etc) -- so far red/green
#include the name of stock in the grpah
#make the ui more visually appealing

#requires another thing to make FUTURE predictions
#the graph looking shifted has to do with the lstm model itself.. think of changing it

# https://stackoverflow.com/questions/70420155/how-to-predict-actual-future-values-after-testing-the-trained-lstm-model
with open('all_tickers.txt', 'r') as file:
    stock_symbols = file.read().splitlines()

# Streamlit App
def main():
    # global predictor
    st.title("Stock Price Predictor")
    StockSymbol = st.sidebar.text_input("Enter Stock Symbol (not name)", "")

    # Check if the input is valid
    if StockSymbol:
        if StockSymbol in stock_symbols:
            st.success(f"Stock symbol '{StockSymbol}' is valid!")
            # st.write("You can proceed with your analysis.")
            predictor = StockPredictor(StockSymbol)
            predictor.setStockName()
            st.sidebar.text(f"{predictor.StockName} Stock")
            st.write(f"Running model for {predictor.StockName} stocks")
        else:
            st.error("Invalid stock symbol. Please enter a valid one.")
            # Prevent further actions
            st.stop()
 
    if st.sidebar.button("Run Predictor"):
        if StockSymbol is None:
            st.error("Please input a Stock Symbol to predict.")
            return  
       
        X_test, y_test = predictor.run()
        predictor.predict(X_test, y_test)
        st.write(f"Prediction Accuracy: {predictor.accuracy:.2f}%")
       
        # Display predictions
        st.line_chart({
            "Actual Prices": predictor.y_test,
            "Predicted Prices": predictor.predicted_price
            }, color= ((255, 0, 0), (0, 255, 0)),
            x_label="Date",
            y_label="Price")
   
if __name__ == "__main__":
    main()

