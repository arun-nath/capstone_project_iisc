import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import time
from sqlalchemy import create_engine
import os

NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "HDFCBANK", "ICICIBANK", 
    "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "BAJFINANCE", "ASIANPAINT", 
    "DMART", "MARUTI", "AXISBANK", "SUNPHARMA", "TITAN", "ULTRACEMCO", 
    "TATAMOTORS", "NESTLEIND", "BAJAJFINSV", "JSWSTEEL", "POWERGRID", 
    "TATASTEEL", "ADANIPORTS", "HCLTECH", "WIPRO", "DRREDDY", "CIPLA", 
    "HDFCLIFE", "ONGC", "NTPC", "TECHM", "INDUSINDBK", "COALINDIA", 
    "GRASIM", "TATACONSUM", "SBILIFE", "BRITANNIA", "HEROMOTOCO", "EICHERMOT", 
    "DIVISLAB", "BAJAJ-AUTO", "SHREECEM", "UPL", "APOLLOHOSP", "HINDALCO", 
    "VEDL", "BPCL"
]

def fetch_nifty50_data():
    """
    Fetch 5-minute OHLCV data for all Nifty 50 stocks
    """
    all_data = []
    failed_symbols = []
    
    for i, symbol in enumerate(NIFTY_50_SYMBOLS):
        try:
            # Add delay to avoid rate limiting (0.5 seconds between requests)
            if i > 0:
                time.sleep(0.5)
            
            yahoo_symbol = f"{symbol}.NS"
            print(f"Fetching data for {yahoo_symbol} ({i+1}/{len(NIFTY_50_SYMBOLS)})")
            
            # Fetch 5-minute data for last 1 day
            stock = yf.Ticker(yahoo_symbol)
            data = stock.history(period='1d', interval='5m')
            
            if data.empty or len(data) < 2:
                print(f"No data or insufficient data for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Get the latest complete candle
            latest_candle = data.iloc[-1]
            
            
            stock_data = {
                'timestamp': data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                'open': round(float(latest_candle['Open']), 2),
                'high': round(float(latest_candle['High']), 2),
                'low': round(float(latest_candle['Low']), 2),
                'close': round(float(latest_candle['Close']), 2),
                'volume': int(latest_candle['Volume']),
                'symbol':symbol 
            }
            
            all_data.append(stock_data)
            print(f"✅ Success: {symbol} - ₹{stock_data['close']}")
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
            failed_symbols.append(symbol)
    
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df = fetch_nifty50_data()
    db_user = os.environ.get("DB_USERNAME")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = 'database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com'
    db_port = '5432'
    db_name = 'capstone_project'
    print(df)

    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    table_name = 'stocks'
    df.to_sql(table_name, engine, if_exists='append', index=False)