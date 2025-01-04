import time
import requests
import yfinance as yf
import json

# Step 1: Get the stock symbols from the file
url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
response = requests.get(url)
stock_symbols = response.text.splitlines()

# Step 2: Fetch stock names using yfinance with a delay
stock_dict = {}
for symbol in stock_symbols:  # Limit to the first 10 for testing
    try:
        print(symbol)
        stock = yf.Ticker(symbol)
        stock_name = stock.info.get('shortName', 'Unknown')
        stock_dict[symbol] = stock_name
        # Add a delay to avoid rate limits
        time.sleep(1)  # Adjust the time to balance rate limit and performance
    except Exception as e:
        stock_dict[symbol] = 'Error fetching name'

# Save the resulting dictionary to a JSON file
with open('stock_dict.json', 'w') as json_file:
    json.dump(stock_dict, json_file, indent=4)

print("Dictionary saved to 'stock_dict.json'")
