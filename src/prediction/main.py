from tensorflow import keras
import pandas as pd
import numpy as np

from sqlalchemy import create_engine, text
from sklearn.preprocessing import RobustScaler, LabelEncoder

import os

"""
To DO

Retun the model for all 50 stocks
Save scaler and encoders
Save the data to table
"""



def calculate_features(df):
    """
    Calculate features for multiple stocks in the same dataframe
    """
    # Make sure we have the required columns
    required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # df = df.reset_index()
    # Sort by symbol and datetime
    # df = df.sort_values(['symbol', 'timestamp']).copy()

    features_list = []

    # Process each stock separately
    for symbol, group in df.groupby('symbol'):
        symbol_df = group.copy()

        # Calculate returns
        symbol_df['returns'] = symbol_df['close'].pct_change()
        symbol_df['returns_5']  = symbol_df['close'].pct_change(5)
        symbol_df['log_returns'] = np.log(symbol_df['close']).diff()
        symbol_df['prev_close'] = symbol_df['close'].shift(1)
        symbol_df['vwap'] = (symbol_df['volume'] * (symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 3).cumsum() / symbol_df['volume'].cumsum()
        symbol_df['gap'] = (symbol_df['open'] - symbol_df['prev_close']) / symbol_df['prev_close'] * 100
        symbol_df['gap_abs'] = abs(symbol_df['gap'])
        symbol_df['gap_direction'] = np.where(symbol_df['gap'] > 0, 1, np.where(symbol_df['gap'] < 0, -1, 0))
        symbol_df['ema_8'] = symbol_df['close'].ewm(span=8).mean()
        symbol_df['ema_15'] = symbol_df['close'].ewm(span=15).mean()
        ema12 = symbol_df['close'].ewm(span=12, adjust=False).mean()
        ema26 = symbol_df['close'].ewm(span=26, adjust=False).mean()
        symbol_df['macd'] = ema12 - ema26
        symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9, adjust=False).mean()
        symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']
        for n in [5, 10, 20]:
         symbol_df[f'roc_{n}'] = symbol_df['close'].pct_change(periods=n)

        df['up_streak'] = df['close'].gt(df['close'].shift()).astype(int)
        df['up_streak'] = df['up_streak'] * (df['up_streak'].groupby((df['up_streak'] != df['up_streak'].shift()).cumsum()).cumsum())


        # Calculate up moves and down moves for RSI calculation
        symbol_df['up_move'] = np.where(symbol_df['close'] > symbol_df['close'].shift(1),
                                       symbol_df['close'] - symbol_df['close'].shift(1), 0)
        symbol_df['down_move'] = np.where(symbol_df['close'] < symbol_df['close'].shift(1),
                                         symbol_df['close'].shift(1) - symbol_df['close'], 0)

        symbol_df["sma_8"] = symbol_df["close"].rolling(window=8).mean()
        symbol_df["sma_20"] = symbol_df["close"].rolling(window=20).mean()

        symbol_df['sma8_above_sma20'] = (symbol_df['sma_8'] > symbol_df['sma_20']).astype(int)
        symbol_df['vwap_sma_20'] = symbol_df['vwap'].rolling(window=20).mean()
        symbol_df['vwap_ratio'] = symbol_df['vwap'] / symbol_df['vwap_sma_20']

        # --- VOLATILITY & RISK FEATURES ---
        # Calculate ATR (Average True Range)
        symbol_df['tr0'] = symbol_df['high'] - symbol_df['low']
        symbol_df['tr1'] = abs(symbol_df['high'] - symbol_df['close'].shift(1))
        symbol_df['tr2'] = abs(symbol_df['low'] - symbol_df['close'].shift(1))
        symbol_df['tr'] = symbol_df[['tr0', 'tr1', 'tr2']].max(axis=1)
        symbol_df['atr'] = symbol_df['tr'].rolling(20).mean()

        symbol_df['volatility_20'] = symbol_df['returns'].rolling(20).std()
        symbol_df['volatility_50'] = symbol_df['returns'].rolling(50).std()
        symbol_df['volatility_ratio'] = symbol_df['volatility_20'] / symbol_df['volatility_50']
        symbol_df['atr_pct'] = symbol_df['atr'] / symbol_df['close']

        # --- PRICE & MOMENTUM FEATURES ---
        # Trend Strength
        for window in [10, 20, 30]:
            symbol_df[f'momentum_{window}'] = symbol_df['close'] / symbol_df['close'].shift(window) - 1

        # Acceleration & Rate of Change
        symbol_df['momentum_roc_10'] = symbol_df['momentum_10'] - symbol_df['momentum_20']

        # RSI Calculation
        symbol_df['rsi_14'] = 100 - (100 / (1 + (symbol_df['up_move'].rolling(14).mean() /
                                               symbol_df['down_move'].rolling(14).mean().replace(0, 0.001))))

        # Bollinger Bands
        bb_window = 20
        symbol_df['bb_mid'] = symbol_df['close'].rolling(bb_window).mean()
        bb_std = symbol_df['close'].rolling(bb_window).std()
        symbol_df['bb_upper'] = symbol_df['bb_mid'] + (bb_std * 2)
        symbol_df['bb_lower'] = symbol_df['bb_mid'] - (bb_std * 2)
        symbol_df['price_vs_bb'] = (symbol_df['close'] - symbol_df['bb_mid']) / (symbol_df['bb_upper'] - symbol_df['bb_lower'])

        # --- VOLUME FEATURES ---
        symbol_df['volume_ma_20'] = symbol_df['volume'].rolling(20).mean()
        symbol_df['volume_ma_50'] = symbol_df['volume'].rolling(50).mean()
        symbol_df['volume_zscore'] = (symbol_df['volume'] - symbol_df['volume_ma_50']) / symbol_df['volume'].rolling(50).std()

        # Volume-Price Confirmation
        symbol_df['volume_price_correlation'] = symbol_df['volume'].rolling(20).corr(symbol_df['close'])

        # On-Balance Volume (OBV)
        symbol_df['obv'] = (np.sign(symbol_df['returns']) * symbol_df['volume']).cumsum()
        symbol_df['obv_ma_ratio'] = symbol_df['obv'] / symbol_df['obv'].rolling(50).mean()

        # --- TIME FEATURES ---
        symbol_df['hour'] = symbol_df['timestamp'].dt.hour
        symbol_df['minute'] = symbol_df['timestamp'].dt.minute
        symbol_df['is_first_hour'] = ((symbol_df['hour'] == 9) |
                                    ((symbol_df['hour'] == 10) & (symbol_df['minute'] < 15))).astype(int)
        symbol_df['is_last_hour'] = (symbol_df['hour'] >= 15).astype(int)

        symbol_df['hl_range_pct'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['close'] * 100
        symbol_df['upper_shadow'] = symbol_df['high'] - symbol_df[['open','close']].max(axis=1)
        symbol_df['lower_shadow'] = symbol_df[['open','close']].min(axis=1) - symbol_df['low']

        # --- LAGGED RETURNS ---
        for lag in [1, 2, 5, 10]:
            symbol_df[f'return_lag_{lag}'] = symbol_df['returns'].shift(lag)

        # Add symbol back to identify which stock this belongs to
        symbol_df['symbol'] = symbol

        features_list.append(symbol_df)


    # Combine all stocks
    df_with_features = pd.concat(features_list, ignore_index=True)
    # Drop all rows with NaN values created by rolling calculations
    df_clean = df_with_features.dropna()

    return df_clean

def create_sequences(data, time_steps=30):
    """Convert tabular data to time series sequences for LSTM"""
    X_seq = []
    for i in range(time_steps, len(data)):
        X_seq.append(data[i-time_steps:i])
    return np.array(X_seq)

def get_data(datetime, symbol):
    db_user = os.environ.get("DB_USERNAME")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = 'database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com'
    db_port = '5432'
    db_name = 'capstone_project'

    # 3. Create a SQLAlchemy engine
    # The 'postgresql+psycopg2://' dialect specifies the use of psycopg2
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    with engine.connect() as conn:
        query = f"select * from stocks where timestamp<'{datetime}' and symbol='{symbol}' order by timestamp desc limit 120"
        df = pd.read_sql_query(text(query), conn,  parse_dates=['timestamp'])
        df = df.iloc[::-1].reset_index(drop=True)
    return df

if __name__ == "__main__":
    model = keras.models.load_model("lstm_1hr_prediction_2_atr_multiplier_all_stocks.keras")
    datetime = '2025-08-08 15:00:00'
    SYMBOL_LIST =  [
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
    
    final_prediction = []
    for symbol in SYMBOL_LIST:
        symbol_pred = {}
        df = get_data(datetime, symbol)
        df.index = df['timestamp']
        categorical_cols = ['symbol']
        feature_columns = [
            'returns', 'volatility_20', 'volatility_50', 'volatility_ratio', 'atr_pct',
            'momentum_10', 'momentum_20', 'momentum_30', 'momentum_roc_10', 'rsi_14',
            'price_vs_bb', 'volume_ma_20', 'volume_ma_50', 'volume_zscore',
            'volume_price_correlation', 'obv_ma_ratio', 'sma_8', 'sma_20', 'sma8_above_sma20',
            'vwap', 'gap', 'gap_direction', 'ema_8', 'ema_15', 'vwap_sma_20',
            'vwap_ratio', 'returns_5', 'log_returns', 'hl_range_pct',
            'upper_shadow', 'lower_shadow','macd','macd_signal','macd_hist','roc_5','roc_10','roc_20'
        ] + categorical_cols
        df_features = calculate_features(df)
        df_features = df_features[feature_columns]
        num_cols = [c for c in feature_columns if c not in categorical_cols]
        scaler = RobustScaler()
        df_features[num_cols] = scaler.fit_transform(df_features[num_cols])  ##
        for col in categorical_cols:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))   
        df_seq = create_sequences(df_features)
        y_test_pred_proba_lstm = model.predict(df_seq).flatten()
        symbol_pred['timestamp'] = datetime
        symbol_pred['symbol'] = symbol
        symbol_pred['buy_pred'] = round(y_test_pred_proba_lstm[-1],2)
        symbol_pred['sell_pred'] = round((1-y_test_pred_proba_lstm[-1]), 2)
        # symbol_pred[symbol] = (round(y_test_pred_proba_lstm[-1],2),round((1-y_test_pred_proba_lstm[-1]), 2))
        final_prediction.append(symbol_pred)
    print(final_prediction)
    df_result = pd.DataFrame(final_prediction)
    print(df_result)

    



