# Use a custom image for Python 3.13.0 if available
FROM python:3.13.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Streamlit project files into the container
COPY Project1_LSTMSandBasicML /app/Project1_LSTMSandBasicML

# Copy the requirements.txt file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Project1_LSTMSandBasicML/StockPredictorUIApp.py", "--server.port=8501", "--server.enableCORS=false"]
