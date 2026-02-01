import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ===============================
# 1. DOWNLOAD DATA
# ===============================
SYMBOL = "AAPL"        # change to MSFT, GOOGL, TSLA, etc.
PERIOD = "5y"          # best for LSTM
WINDOW_SIZE = 60

data = yf.download(SYMBOL, period=PERIOD)
close_prices = data[["Close"]]

# ===============================
# 2. SCALE DATA
# ===============================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ===============================
# 3. CREATE SEQUENCES
# ===============================
X, y = [], []

for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# ===============================
# 4. TRAIN / TEST SPLIT
# ===============================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================
# 5. BUILD LSTM MODEL (CLEAN)
# ===============================
model = Sequential([
    Input(shape=(WINDOW_SIZE, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# ===============================
# 6. TRAIN MODEL
# ===============================
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ===============================
# 7. PREDICT NEXT DAY PRICE
# ===============================
last_60_days = scaled_data[-WINDOW_SIZE:]
last_60_days = last_60_days.reshape(1, WINDOW_SIZE, 1)

predicted_scaled = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

last_price = close_prices.iloc[-1, 0]

trend = "UP 📈" if predicted_price > last_price else "DOWN 📉"

print("\n===============================")
print(f"Stock: {SYMBOL}")
print(f"Last Close Price: {last_price:.2f}")
print(f"Predicted Next Price: {predicted_price:.2f}")
print(f"Predicted Trend: {trend}")
print("===============================\n")

# ===============================
# 8. VISUALIZE RESULTS
# ===============================
predicted_test = model.predict(X_test)
predicted_test = scaler.inverse_transform(predicted_test)

real_prices = close_prices.iloc[split + WINDOW_SIZE:].values

plt.figure(figsize=(12, 6))
plt.plot(real_prices, color="blue", label="Actual Price")
plt.plot(predicted_test, color="red", label="Predicted Price")
plt.title(f"{SYMBOL} Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
