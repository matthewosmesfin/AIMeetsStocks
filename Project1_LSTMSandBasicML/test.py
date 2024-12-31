import streamlit as st
import pandas as pd
from LSTMMultiFeaure import StockPredictor  # Assuming the class is saved as stock_predictor.py


# Streamlit App
def main():
    # global predictor
    st.title("Stock Price Predictor (w/ Training and Testing)")
    # Initialize Predictor
    if "predictor" not in st.session_state:
        predictor = StockPredictor("Google")
        st.session_state.predictor = predictor
        st.info("Predictor is initaliazed")
    else:
        predictor = st.session_state.predictor
        st.info("Skipping Predictor initalization")
    st.sidebar.header("Settings")
   
    predictor.StockName = st.sidebar.selectbox("Select Stock", ["Google", "Tesla"]) #make it able
    train_file = st.sidebar.file_uploader("Upload First File (Training or Training/Testing)", type=["csv"])
    test_file = st.sidebar.file_uploader("Upload Additional File (Testing only)", type=["csv"])
         
    predictor.TrainPercent = st.sidebar.slider("Training Percentage", 0.5, 1.0, 0.8)
    predictor.ValidationPercent = st.sidebar.slider("Validation Percentage", 0.0, 0.5, 0.2)
    predictor.TimeStep = st.sidebar.number_input("Time Step", min_value=1, value=60)
    predictor.BatchSize = st.sidebar.number_input("Batch Size", min_value=1, value=32, step=1)
    predictor.Dropout = st.sidebar.number_input("Dropout", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    predictor.Units = st.sidebar.number_input("Units", min_value=1, value=50, step=1)
    predictor.Retune = st.sidebar.checkbox("Enable Hyperparameter Tuning")
    predictor.SkipTraining = st.sidebar.checkbox("Skip Training and Use the existing Model")
   
 
    if st.sidebar.button("Run Predictor"):
        if train_file is None:
            st.error("Please upload a training file.")
            return  
       
        X_test, y_test = predictor.run(train_file, test_file)
        predictor.predict(X_test, y_test)
        st.write(f"Prediction Accuracy: {predictor.accuracy:.2f}%")
       
        # Display predictions
        st.line_chart({
            "Actual Prices": [y[0] for y in predictor.y_test],
            "Predicted Prices": [y[0] for y in predictor.predicted_price]
        })
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

        # predictor.BatchSize = 16
        # predictor.Dropout = 0.3
        # predictor.Units = 50
       
        #Train the model
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

