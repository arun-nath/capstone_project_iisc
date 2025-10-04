
import pandas as pd
import json
from datetime import datetime, timedelta
from datetime import datetime, time as dt_time
import time
from sqlalchemy import create_engine
import yfinance as yf
import pytz

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

holiday_list = [
'2025-01-26',
'2025-02-26',
'2025-03-14',
'2025-03-31',
'2025-04-06',
'2025-04-10',
'2025-04-14',
'2025-04-18',
'2025-05-01',
'2025-06-07',
'2025-07-06',
'2025-08-15',
'2025-08-27',
'2025-10-02',
'2025-10-21',
'2025-10-22',
'2025-11-05',
'2025-12-25',
]

market_start_time = "9:15"
market_end_time = "3:30"

def is_market_time():
    # Convert current time to IST
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist).time()
    now_date = datetime.now(ist).date().strftime('%Y-%m-%d')

    # Define NSE market hours
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)

    return market_open <= now_ist <= market_close and now_date not in holiday_list

def fetch_nifty50_data():
    """
    Fetch 5-minute OHLCV data for all Nifty 50 stocks
    """
    all_data = []
    failed_symbols = []

    if not is_market_time():
        print("Outside market hours")
        return None
    
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

# if __name__ == "__main__":
def lambda_handler(event, context):
    df = fetch_nifty50_data()
    if df is not None:
        db_user = 'postgres'
        db_password = os.environ.get("DB_PASSWORD")
        db_host = 'database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com'
        db_port = '5432'
        db_name = 'capstone_project'
        print(df)

        engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
        table_name = 'stocks'
        df.to_sql(table_name, engine, if_exists='append', index=False)