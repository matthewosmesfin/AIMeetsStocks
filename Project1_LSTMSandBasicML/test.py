import streamlit as st
import pandas as pd
from LSTM_MultiFeaure import StockPredictor  # Assuming the class is saved as stock_predictor.py

# Streamlit App
def main():
    st.title("Stock Price Predictor")
    st.sidebar.header("Settings")
    
    stock_name = st.sidebar.selectbox("Select Stock", ["Google"])
    train_file = st.sidebar.file_uploader("Upload Training Data", type=["csv"])
    test_file = st.sidebar.file_uploader("Upload Testing Data", type=["csv"])
    
    if st.sidebar.button("Run Predictor"):
        if train_file is None:
            st.error("Please upload a training file.")
            return

        # Load data
        #train_data = pd.read_csv(train_file)
        #test_data = pd.read_csv(test_file) if test_file else None
        
        # Initialize Predictor
        predictor = StockPredictor(stock_name)
        
        # Load and preprocess data
        predictor.readData(train_file, test_file if test_file else None)
        st.write("Data loaded and preprocessed successfully.")
        
        # Process training data
        try:
            X_train, y_train = predictor.processTrain(predictor.TrainData.iloc[:, 1:5].values)
            st.write("Training data processed.")
        except Exception as e:
            st.error(f"Error in processing training data: {e}")
            return
        
        # # Hyperparameter tuning (PASS FOR NOW)
        # st.write("Tuning hyperparameters...")
        # try:
        #     predictor.hyperParameterTuning(X_train, y_train, features=4) #issue with this funciton
        #     st.success("Hyperparameter tuning completed.")
        # except Exception as e:
        #     st.error(f"Error during hyperparameter tuning: {e}")
        #     return

        predictor.BatchSize = 16
        predictor.Dropout = 0.3
        predictor.Units = 50
        
        # Train the model
        st.write("Training the model...")
        try:
            predictor.defineAndTrainModel(X_train, y_train, inputs=4, epochs=15) #for now
            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Error during model training: {e}")
            return
        
        # Process test data and predict
        if test_file is not None:
            X_test, y_test = predictor.processTest(predictor.TestData.iloc[:,1:5])
            predictor.predict(X_test, y_test)
            st.write(f"Prediction Accuracy: {predictor.accuracy:.2f}%")
            
            # Display predictions
            st.line_chart({
                "Actual Prices": [y[0] for y in predictor.y_test],
                "Predicted Prices": [y[0] for y in predictor.predicted_price]
            })
    
if __name__ == "__main__":
    main()
