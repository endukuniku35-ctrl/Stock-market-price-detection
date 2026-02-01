from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
model = load_model("model/lstm_model.keras")

SECTORS = {
    "Technology": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google"},
    "Automobile": {"TSLA": "Tesla", "F": "Ford", "GM": "General Motors"},
    "Energy": {"XOM": "Exxon", "CVX": "Chevron"},
    "Finance": {"JPM": "JPMorgan", "BAC": "Bank of America"}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stocks")
def stocks():
    return jsonify(SECTORS)

@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.json["symbol"]

    df = yf.download(symbol, period="10y", interval="1d", progress=False)
    close = df["Close"].astype(float).values

    prices = [round(float(x), 2) for x in close]
    dates = df.index.strftime("%Y-%m-%d").tolist()

    current_price = prices[-1]
    previous_price = prices[-2]

    trend = "UP 📈" if current_price > previous_price else "DOWN 📉"

    future_7 = round(current_price * 1.02, 2)
    future_30 = round(current_price * 1.05, 2)

    summary = {
        "past": f"5Y average price: ${round(np.mean(prices[:len(prices)//2]), 2)}",
        "present": f"Current price: ${current_price}",
        "future": f"Expected range: ${future_7} – ${future_30}"
    }

    return jsonify({
        "dates": dates[-200:],
        "prices": prices[-200:],
        "current_price": current_price,
        "trend": trend,
        "future_7": future_7,
        "future_30": future_30,
        "summary": summary
    })

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    data = request.json
    filename = "stock_report.pdf"

    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 40, "Stock Prediction Report")

    c.setFont("Helvetica", 12)
    y = h - 80
    for k, v in data.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 25

    c.save()
    return jsonify({"status": "PDF generated"})

if __name__ == "__main__":
    app.run(debug=True)
