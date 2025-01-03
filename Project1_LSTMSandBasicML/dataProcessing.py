import requests
import json
import pandas as pd

#Data Processing 

KEY = "ZCAWN7H87AEQHGDD"
SYMBOL = "IBM"

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={KEY}'
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


# for key,value in data["Time Series (Daily)"].items():
#     print(f"{SYMBOL} Open value for {key} is {value[o]}")
#     print(f"{SYMBOL} High value for {key} is {value[h]}")
#     print(f"{SYMBOL} Low value for {key} is {value[l]}")
#     print(f"{SYMBOL} Close value for {key} is {value[c]}")

for key,value in data["Time Series (Daily)"].items():
    Dates.append(key)
    Open.append(value[o])
    High.append(value[h])
    Low.append(value[l])
    Close.append(value[c])

stockdf = pd.DataFrame({
    "Open": Open,
    "High": High,
    "Low": Low,
    "Close": Close
}, index=pd.to_datetime(Dates))

print(len(stockdf))

