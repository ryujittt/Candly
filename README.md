# Candly
Candly is a Python application that visualizes and analyzes (OHLCV) data. It compares recent market patterns to historical data using similarity metrics and basic machine learning to provide predictive insights, it aims to assist traders and researchers in recognizing patterns that precede similar future behavior.

# Features
Load historical OHLCV data from CSV files.

Fetch real-time data using the Binance API via ccxt.

Perform candlestick pattern similarity analysis using:

Cosine similarity

Cross-correlation

Dynamic Time Warping (DTW)

Pearson correlation

Derivatives and standard deviation comparisons

Apply K-Nearest Neighbors (KNN) to find the closest historical match.

Visualize both the current pattern and its closest historical match with candlestick charts.

 # Installation
Make sure you have Python 3.7+ installed.
Required packages include:

PyQt5,matplotlib,pandas,numpy,ccxt,scipy,sklearn,fastdtw,mplfinance

 #Usage

Select a symbol (e.g., BTC/USDT) and timeframe (5m, 1h, 1d, etc.).

Set the number of candles for pattern comparison (past) and forecast (future).

- Load historical data from CSVs .
- load current data
- process data
- correlate 
- iterate through correlatins using next - last buttons
- find the best fit using a simple machine learning approach (KNeighborsClassifier)

# ⚠️ Disclaimer
This software is intended for educational and research purposes only. No financial advice is provided. Use at your own risk.

Candly Project. All rights reserved.

Open-source code, developed by [CHAKRAR ABDELMALIK].




