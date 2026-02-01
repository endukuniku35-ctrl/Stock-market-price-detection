import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os

os.makedirs("model", exist_ok=True)

symbol = "AAPL"
df = yf.download(symbol, period="10y", interval="1d")

close = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

model = Sequential([
    Input(shape=(60, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=10, batch_size=32)

model.save("model/lstm_model.keras")

print("✅ Model trained & saved")
