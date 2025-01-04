import requests
import pandas as pd

#Data Processing 

with open('/Users/matthewosmesfin/Documents/apikey', 'r') as file:
    KEY = file.read()
SYMBOL = "AX"

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

print(stockdf.head)