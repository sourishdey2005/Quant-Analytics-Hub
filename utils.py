"""
utils.py - Comprehensive Stock Analysis & ML Prediction Utilities
Complete feature engineering, ML model integration, and data processing
Optimized for NIFTY-50 dataset with columns: Date, Symbol, Series, Prev Close, Open, High, Low, Last, Close, VWAP
"""
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from scipy.stats import linregress, zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# MODULE 0: CORE CONFIGURATION & CONSTANTS
# ==================================================

# Basic price columns required
BASIC_PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Prev Close', 'VWAP', 'Last']
TECHNICAL_WINDOWS = [5, 10, 20, 50, 100, 200]
VOLATILITY_WINDOWS = [10, 20, 50]
MA_WINDOWS = [20, 50, 100, 200]

# ML Model paths (update these paths as needed)
MODEL_CONFIG = {
    'direction_model': 'nifty50_quant_models_and_data/direction_model.pkl',
    'return_model': 'nifty50_quant_models_and_data/return_model.pkl',
    'scaler': 'nifty50_quant_models_and_data/scaler.pkl',
    'symbol_encoder': 'nifty50_quant_models_and_data/symbol_encoder.pkl'
}

# Decision thresholds
DECISION_THRESHOLDS = {
    'strong_buy': 0.7,
    'buy': 0.4,
    'accumulate': 0.2,
    'hold': -0.1,
    'reduce': -0.3,
    'sell': -0.6,
    'strong_sell': -1.0
}

RISK_LEVELS = {
    'HIGH': 3,
    'MEDIUM': 2,
    'LOW': 1
}

# ==================================================
# MODULE 1: DATA LOADING & PREPROCESSING
# ==================================================

def load_data(filepath_or_buffer):
    """
    Load NIFTY-50 data from file path or uploaded buffer
    Handles CSV, Excel files with robust error handling
    """
    try:
        df = None
        
        # Handle file buffer (Streamlit upload)
        if hasattr(filepath_or_buffer, 'read'):
            filepath_or_buffer.seek(0)
            if hasattr(filepath_or_buffer, 'name'):
                if filepath_or_buffer.name.endswith('.csv'):
                    df = pd.read_csv(filepath_or_buffer)
                elif filepath_or_buffer.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filepath_or_buffer)
                else:
                    raise ValueError("Unsupported file format. Use CSV or Excel.")
            else:
                df = pd.read_csv(filepath_or_buffer)
        
        # Handle file path string
        elif isinstance(filepath_or_buffer, str):
            if filepath_or_buffer.endswith('.csv'):
                df = pd.read_csv(filepath_or_buffer)
            elif filepath_or_buffer.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath_or_buffer)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        if df is None:
            raise ValueError("Could not load data from provided source")
        
        # Validate required columns
        required_cols = ['Date', 'Symbol', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try case-insensitive matching
            for col in missing_cols.copy():
                col_lower = col.lower()
                matching_cols = [c for c in df.columns if c.lower() == col_lower]
                if matching_cols:
                    df.rename(columns={matching_cols[0]: col}, inplace=True)
                    missing_cols.remove(col)
            
            if missing_cols:
                raise ValueError(f"Required columns missing: {missing_cols}")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Symbol', 'Close'])
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Prev Close', 'VWAP', 'Last']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing price values using forward fill within symbol groups
        price_cols = ['Open', 'High', 'Low', 'Close', 'Prev Close', 'VWAP', 'Last']
        for col in price_cols:
            if col in df.columns:
                df[col] = df.groupby('Symbol')[col].transform(
                    lambda x: x.ffill().bfill()
                )
        
        # Create Prev Close if missing
        if 'Prev Close' not in df.columns:
            df['Prev Close'] = df.groupby('Symbol')['Close'].shift(1)
            df['Prev Close'] = df.groupby('Symbol')['Prev Close'].transform(
                lambda x: x.bfill()
            )
        
        # Calculate VWAP if missing
        if 'VWAP' not in df.columns or df['VWAP'].isnull().all():
            df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Add Volume column if missing (for calculations)
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000
        
        print(f"✅ Loaded {len(df)} rows with {len(df['Symbol'].unique())} unique symbols")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading  {str(e)}")
        raise

def validate_data_structure(df):
    """
    Validate that dataframe has correct structure for analysis
    """
    issues = []
    
    # Check required columns
    required = ['Date', 'Symbol', 'Close']
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        issues.append("Date column is not datetime type")
    
    # Check for null values in critical columns
    critical_cols = ['Date', 'Symbol', 'Close']
    for col in critical_cols:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            issues.append(f"Column {col} has {null_count} null values")
    
    # Check date range
    if df['Date'].nunique() < 30:
        issues.append("Insufficient data: Less than 30 unique dates")
    
    # Check symbol count
    if df['Symbol'].nunique() < 1:
        issues.append("No symbols found in data")
    
    return issues

def clean_and_prepare_data(df):
    """
    Clean and prepare data for feature engineering
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
    
    # Sort by symbol and date
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# ==================================================
# MODULE 2: BASIC PRICE & TREND FEATURES
# ==================================================

def calculate_basic_price_features(df):
    """
    MODULE 1 — PRICE & TREND INTELLIGENCE
    Calculate fundamental price and trend features
    """
    df = df.copy()
    
    # 1. Returns
    df['Return'] = df.groupby('Symbol')['Close'].pct_change()
    df['Return_Pct'] = df['Return'] * 100
    
    # 2. Price changes
    df['Price_Change'] = df.groupby('Symbol')['Close'].diff()
    
    # 3. Intraday range
    df['Intraday_Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Intraday_Range'] / df['Close'].replace(0, np.nan)) * 100
    
    # 4. Open vs Close spread
    df['Open_Close_Spread'] = df['Close'] - df['Open']
    df['Open_Close_Spread_Pct'] = (df['Open_Close_Spread'] / df['Open'].replace(0, np.nan)) * 100
    
    # 5. Gap analysis
    df['Gap'] = df['Open'] - df['Prev Close']
    df['Gap_Pct'] = (df['Gap'] / df['Prev Close'].replace(0, np.nan)) * 100
    
    # 6. Cumulative returns
    df['CumReturn'] = df.groupby('Symbol')['Return'].apply(
        lambda x: (1 + x.fillna(0)).cumprod() - 1
    ).reset_index(drop=True)
    df['CumReturn_Pct'] = df['CumReturn'] * 100
    
    # 7. Normalized price (base 100)
    df['NormPrice'] = df.groupby('Symbol')['Close'].transform(
        lambda x: (x / x.iloc[0]) * 100 if len(x) > 0 else np.nan
    )
    
    # 8. Price momentum (Δ Close)
    df['Price_Momentum'] = df.groupby('Symbol')['Close'].diff(5)  # 5-day momentum
    
    # 9. Trend slope using rolling linear regression
    def calculate_trend_slope(series, window=20):
        """Calculate slope of price trend using linear regression"""
        if len(series) < window:
            return np.nan
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                x = np.arange(window)
                y = series.iloc[i-window+1:i+1].values
                if np.isnan(y).any():
                    slopes.append(np.nan)
                else:
                    try:
                        slope, _, _, _, _ = linregress(x, y)
                        slopes.append(slope)
                    except:
                        slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    df['Trend_Slope_20D'] = df.groupby('Symbol')['Close'].transform(
        lambda x: calculate_trend_slope(x, window=20)
    )
    
    # 10. Price acceleration (2nd derivative)
    df['Price_Acceleration'] = df.groupby('Symbol')['Trend_Slope_20D'].diff()
    
    # 11. Price regime classification
    def classify_price_regime(slope_series):
        """Classify price regime: Up / Sideways / Down"""
        regimes = []
        for slope in slope_series:
            if pd.isna(slope):
                regimes.append('Undefined')
            elif slope > 0.5:
                regimes.append('Up')
            elif slope < -0.5:
                regimes.append('Down')
            else:
                regimes.append('Sideways')
        return regimes
    
    df['Price_Regime'] = classify_price_regime(df['Trend_Slope_20D'])
    
    # 12. Rolling returns for different periods
    for period in [20, 50]:
        df[f'Rolling_Return_{period}D'] = df.groupby('Symbol')['Close'].transform(
            lambda x: ((x / x.shift(period)) - 1) * 100
        )
    
    return df

# ==================================================
# MODULE 3: MOVING AVERAGES & SIGNALS
# ==================================================

def calculate_moving_averages(df):
    """
    MODULE 2 — MOVING AVERAGES & SIGNALS
    Calculate moving averages and crossover signals
    """
    df = df.copy()
    
    # 1-3. Moving average overlays
    for window in [20, 50, 100, 200]:
        ma_col = f'MA{window}'
        df[ma_col] = df.groupby('Symbol')['Close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # 4-5. MA crossovers
    # MA20-MA50 crossover
    df['MA20_MA50_Crossover'] = np.sign(df['MA20'] - df['MA50'])
    df['MA20_MA50_Crossover_Change'] = df['MA20_MA50_Crossover'].diff()
    df['MA20_Above_MA50'] = (df['MA20'] > df['MA50']).astype(int)
    
    # MA50-MA100 crossover
    df['MA50_MA100_Crossover'] = np.sign(df['MA50'] - df['MA100'])
    df['MA50_MA100_Crossover_Change'] = df['MA50_MA100_Crossover'].diff()
    df['MA50_Above_MA100'] = (df['MA50'] > df['MA100']).astype(int)
    
    # 6-7. Distance from MAs (%)
    for window in [20, 50, 100, 200]:
        ma_col = f'MA{window}'
        dist_col = f'Dist_MA{window}_Pct'
        df[dist_col] = ((df['Close'] - df[ma_col]) / df[ma_col].replace(0, np.nan)) * 100
    
    # 8. MA slope strength
    for window in [20, 50, 100]:
        ma_col = f'MA{window}'
        slope_col = f'MA{window}_Slope'
        df[slope_col] = df.groupby('Symbol')[ma_col].transform(
            lambda x: x.diff(5) / 5  # 5-day slope
        )
    
    # 9. Trend persistence indicator
    def calculate_trend_persistence(series, window=20):
        """Measure how consistently price stays above/below MA"""
        persistence = []
        for i in range(len(series)):
            if i < window:
                persistence.append(np.nan)
            else:
                recent = series.iloc[i-window+1:i+1]
                pct_above = (recent > 0).mean()
                persistence.append(pct_above * 2 - 1)  # Scale to [-1, 1]
        return pd.Series(persistence, index=series.index)
    
    df['Trend_Persistence_20D'] = df.groupby('Symbol')['MA20_Above_MA50'].transform(
        lambda x: calculate_trend_persistence(x, window=20)
    )
    
    # 10. Golden Cross and Death Cross detection
    df['Golden_Cross'] = (
        (df['MA20'] > df['MA50']) & 
        (df.groupby('Symbol')['MA20'].shift(1) <= df.groupby('Symbol')['MA50'].shift(1))
    ).astype(int)
    
    df['Death_Cross'] = (
        (df['MA20'] < df['MA50']) & 
        (df.groupby('Symbol')['MA20'].shift(1) >= df.groupby('Symbol')['MA50'].shift(1))
    ).astype(int)
    
    return df

# ==================================================
# MODULE 4: VOLATILITY & RISK METRICS
# ==================================================

def calculate_volatility_metrics(df):
    """
    MODULE 4 — RISK & VOLATILITY
    Calculate volatility, drawdown, and risk metrics
    """
    df = df.copy()
    
    # 1-2. Rolling volatility (annualized)
    for window in [10, 20, 50]:
        vol_col = f'Volatility_{window}D'
        df[vol_col] = df.groupby('Symbol')['Return'].transform(
            lambda x: x.rolling(window=window, min_periods=5).std() * np.sqrt(252) * 100
        )
    
    # 3. Volatility regime detection
    def detect_volatility_regime(vol_series):
        """Classify volatility regime: Low / Medium / High / Very High"""
        regimes = []
        for vol in vol_series:
            if pd.isna(vol):
                regimes.append('Undefined')
            elif vol < 15:
                regimes.append('Low')
            elif vol < 30:
                regimes.append('Medium')
            elif vol < 50:
                regimes.append('High')
            else:
                regimes.append('Very High')
        return regimes
    
    df['Volatility_Regime'] = detect_volatility_regime(df['Volatility_20D'])
    
    # 4. High-Low volatility proxy
    df['HL_Volatility'] = df.groupby('Symbol')['Range_Pct'].transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )
    
    # 5-6. Drawdown calculations
    df['Rolling_Max'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.expanding().max()
    )
    df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
    df['Drawdown_Pct'] = df['Drawdown'] * 100
    
    # Maximum drawdown over rolling windows
    for window in [20, 50, 100]:
        mdd_col = f'Max_Drawdown_{window}D'
        df[mdd_col] = df.groupby('Symbol')['Drawdown_Pct'].transform(
            lambda x: x.rolling(window=window, min_periods=5).min()
        )
    
    # 7. Average True Range (ATR)
    df['TR1'] = df['High'] - df['Low']
    df['TR2'] = abs(df['High'] - df['Close'].shift(1))
    df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR_14'] = df.groupby('Symbol')['TR'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    df['ATR_Pct'] = (df['ATR_14'] / df['Close'].replace(0, np.nan)) * 100
    
    # 8. Tail risk (extreme loss frequency)
    def calculate_tail_risk(returns, threshold=-0.03):
        """Calculate frequency of extreme losses"""
        tail_risk = []
        window = 60  # 3-month window
        for i in range(len(returns)):
            if i < window:
                tail_risk.append(np.nan)
            else:
                recent_returns = returns.iloc[i-window+1:i+1]
                extreme_losses = (recent_returns < threshold).sum()
                tail_risk.append(extreme_losses / window)
        return pd.Series(tail_risk, index=returns.index)
    
    df['Tail_Risk_60D'] = df.groupby('Symbol')['Return'].transform(calculate_tail_risk)
    
    # 9. Value at Risk (VaR) - 95% confidence
    df['VaR_95_20D'] = df.groupby('Symbol')['Return'].transform(
        lambda x: x.rolling(window=20, min_periods=10).quantile(0.05)
    )
    
    # 10. Conditional Value at Risk (CVaR)
    def calculate_cvar(returns, alpha=0.05):
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) < 10:
            return np.nan
        var = returns.quantile(alpha)
        cvar = returns[returns <= var].mean()
        return cvar
    
    df['CVaR_95_20D'] = df.groupby('Symbol')['Return'].rolling(
        window=20, min_periods=10
    ).apply(calculate_cvar, raw=False).reset_index(level=0, drop=True)
    
    return df

# ==================================================
# MODULE 5: VWAP & INSTITUTIONAL FLOW
# ==================================================

def calculate_vwap_features(df):
    """
    MODULE 5 — VWAP & INSTITUTIONAL FLOW
    Calculate VWAP-based features and institutional flow indicators
    """
    df = df.copy()
    
    # 1. Close vs VWAP time series (already have Close and VWAP)
    
    # 2. VWAP deviation (%)
    df['VWAP_Deviation_Pct'] = ((df['Close'] - df['VWAP']) / df['VWAP'].replace(0, np.nan)) * 100
    
    # 3. VWAP slope trend
    df['VWAP_Slope'] = df.groupby('Symbol')['VWAP'].transform(
        lambda x: x.diff(5) / 5  # 5-day slope
    )
    
    # 4. % time price above VWAP
    df['Price_Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    
    # Rolling % above VWAP
    df['Pct_Time_Above_VWAP_20D'] = df.groupby('Symbol')['Price_Above_VWAP'].transform(
        lambda x: x.rolling(window=20, min_periods=5).mean() * 100
    )
    
    # 5. VWAP breakout signals
    def detect_vwap_breakout(close, vwap, window=20):
        """Detect when price breaks above/below VWAP with momentum"""
        breakout = pd.Series(0, index=close.index)
        for i in range(window, len(close)):
            recent_vwap = vwap.iloc[i-window+1:i+1]
            recent_close = close.iloc[i-window+1:i+1]
            
            # Check if price crossed VWAP with strength
            if (close.iloc[i] > vwap.iloc[i]) and (close.iloc[i-1] <= vwap.iloc[i-1]):
                # Bullish breakout
                breakout.iloc[i] = 1
            elif (close.iloc[i] < vwap.iloc[i]) and (close.iloc[i-1] >= vwap.iloc[i-1]):
                # Bearish breakout
                breakout.iloc[i] = -1
        return breakout
    
    df['VWAP_Breakout'] = df.groupby('Symbol').apply(
        lambda x: detect_vwap_breakout(x['Close'], x['VWAP'])
    ).reset_index(level=0, drop=True)
    
    # 6. Accumulation/Distribution zones
    df['VWAP_Position'] = np.where(
        df['Close'] > df['VWAP'] * 1.02, 'Above Zone',
        np.where(df['Close'] < df['VWAP'] * 0.98, 'Below Zone', 'Neutral Zone')
    )
    
    # 7. Institutional bias indicator
    def calculate_institutional_bias(close, vwap, volume, window=20):
        """Estimate institutional buying/selling pressure"""
        bias = pd.Series(0, index=close.index)
        for i in range(window, len(close)):
            price_above_vwap = (close.iloc[i-window+1:i+1] > vwap.iloc[i-window+1:i+1]).mean()
            volume_weighted = ((close.iloc[i-window+1:i+1] - vwap.iloc[i-window+1:i+1]) * 
                             volume.iloc[i-window+1:i+1]).sum() / volume.iloc[i-window+1:i+1].sum()
            bias.iloc[i] = price_above_vwap * np.sign(volume_weighted)
        return bias
    
    df['Institutional_Bias'] = df.groupby('Symbol').apply(
        lambda x: calculate_institutional_bias(x['Close'], x['VWAP'], x['Volume'])
    ).reset_index(level=0, drop=True)
    
    # 8. VWAP strength score
    df['VWAP_Strength_Score'] = (
        (df['Pct_Time_Above_VWAP_20D'] / 100) * 0.4 +
        (np.tanh(df['VWAP_Slope'] * 10) * 0.3) +
        (np.tanh(df['VWAP_Deviation_Pct'] / 10) * 0.3)
    )
    
    # 9. VWAP bands (Bollinger-like bands around VWAP)
    df['VWAP_Std_20D'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )
    df['VWAP_Upper_Band'] = df['VWAP'] + (df['VWAP_Std_20D'] * 2)
    df['VWAP_Lower_Band'] = df['VWAP'] - (df['VWAP_Std_20D'] * 2)
    
    # 10. VWAP convergence/divergence
    df['VWAP_Convergence'] = abs(df['Close'] - df['VWAP']) / df['VWAP'].replace(0, np.nan)
    df['VWAP_Convergence_Trend'] = df.groupby('Symbol')['VWAP_Convergence'].transform(
        lambda x: x.diff(5)
    )
    
    return df

# ==================================================
# MODULE 6: TECHNICAL INDICATORS
# ==================================================

def calculate_technical_indicators(df):
    """
    Calculate RSI, MACD, Bollinger Bands, and other technical indicators
    """
    df = df.copy()
    
    # 1. RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_14'] = df.groupby('Symbol')['Close'].transform(
        lambda x: calculate_rsi(x, window=14)
    )
    
    # 2. MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.ewm(span=12, adjust=False, min_periods=12).mean()
    )
    df['EMA_26'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df.groupby('Symbol')['MACD'].transform(
        lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # 3. Bollinger Bands
    df['BB_Middle'] = df.groupby('Symbol')['Close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    bb_std = df.groupby('Symbol')['Close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).std()
    )
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'].replace(0, np.nan)) * 100
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'].replace(0, np.nan)) * 100
    
    # 4. Stochastic Oscillator
    def calculate_stochastic(high, low, close, window=14):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=window, min_periods=1).min()
        highest_high = high.rolling(window=window, min_periods=1).max()
        k = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
        d = k.rolling(window=3, min_periods=1).mean()
        return k, d
    
    stoch_k, stoch_d = calculate_stochastic(df['High'], df['Low'], df['Close'], window=14)
    df['Stochastic_K'] = stoch_k
    df['Stochastic_D'] = stoch_d
    
    # 5. Williams %R
    def calculate_williams_r(high, low, close, window=14):
        """Calculate Williams %R indicator"""
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        wr = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100
        return wr
    
    df['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'], window=14)
    
    # 6. ADX (Average Directional Index) - simplified
    df['+DM'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0), 0
    )
    df['-DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
    )
    
    df['+DI'] = df.groupby('Symbol')['+DM'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    df['-DI'] = df.groupby('Symbol')['-DM'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    
    df['ADX'] = df.groupby('Symbol').apply(
        lambda x: abs(x['+DI'] - x['-DI']) / (x['+DI'] + x['-DI'] + 1e-10) * 100
    ).reset_index(level=0, drop=True)
    
    # 7. CCI (Commodity Channel Index)
    def calculate_cci(high, low, close, window=20):
        """Calculate CCI indicator"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window, min_periods=1).mean()
        mean_deviation = typical_price.rolling(window=window, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
        return cci
    
    df['CCI'] = calculate_cci(df['High'], df['Low'], df['Close'], window=20)
    
    # 8. ATR-based volatility
    df['ATR_Ratio'] = df['ATR_14'] / df['Close'].replace(0, np.nan)
    
    # 9. Momentum indicators
    for period in [10, 20, 50]:
        df[f'Momentum_{period}D'] = df.groupby('Symbol')['Close'].transform(
            lambda x: x - x.shift(period)
        )
        df[f'ROC_{period}D'] = df.groupby('Symbol')['Close'].transform(
            lambda x: ((x / x.shift(period)) - 1) * 100
        )
    
    # 10. Ultimate Oscillator
    bp = df['Close'] - np.minimum(df['Low'], df['Close'].shift(1))
    tr = df['TR']
    
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    
    df['Ultimate_Oscillator'] = (4 * avg7 + 2 * avg14 + avg28) / 7 * 100
    
    return df

# ==================================================
# MODULE 7: SUPPORT & RESISTANCE
# ==================================================

def calculate_support_resistance(df):
    """
    Calculate support, resistance, and breakout levels
    """
    df = df.copy()
    
    # 1-2. Support and resistance levels (rolling highs/lows)
    for window in [20, 50, 100]:
        df[f'Resistance_{window}D'] = df.groupby('Symbol')['High'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        df[f'Support_{window}D'] = df.groupby('Symbol')['Low'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
    
    # 3. Breakout detection
    def detect_breakout(close, resistance, support, window=5):
        """Detect price breakouts above resistance or below support"""
        breakout = pd.Series(0, index=close.index)
        for i in range(len(close)):
            if i < window:
                continue
            
            # Check for resistance breakout
            if (close.iloc[i] > resistance.iloc[i]) and \
               (close.iloc[i-window:i].max() <= resistance.iloc[i]):
                breakout.iloc[i] = 1  # Bullish breakout
            
            # Check for support breakdown
            elif (close.iloc[i] < support.iloc[i]) and \
                 (close.iloc[i-window:i].min() >= support.iloc[i]):
                breakout.iloc[i] = -1  # Bearish breakdown
        
        return breakout
    
    df['Breakout_20D'] = df.groupby('Symbol').apply(
        lambda x: detect_breakout(x['Close'], x['Resistance_20D'], x['Support_20D'])
    ).reset_index(level=0, drop=True)
    
    # 4. Range compression indicator
    df['Range_Compression_Ratio'] = df['BB_Width'] / df['BB_Width'].rolling(50).mean()
    
    # 5. Distance to nearest support/resistance
    df['Dist_to_Nearest_SR'] = np.minimum(
        abs(df['Close'] - df['Resistance_20D']),
        abs(df['Close'] - df['Support_20D'])
    )
    df['Dist_to_Nearest_SR_Pct'] = (df['Dist_to_Nearest_SR'] / df['Close']) * 100
    
    # 6. Support/Resistance strength (how many times tested)
    def calculate_sr_strength(prices, levels, window=50):
        """Calculate how many times price has tested a level"""
        strength = pd.Series(0, index=prices.index)
        for i in range(window, len(prices)):
            level = levels.iloc[i]
            nearby = abs(prices.iloc[i-window:i] - level) / level < 0.02  # Within 2%
            strength.iloc[i] = nearby.sum()
        return strength
    
    df['Resistance_Strength_20D'] = df.groupby('Symbol').apply(
        lambda x: calculate_sr_strength(x['Close'], x['Resistance_20D'])
    ).reset_index(level=0, drop=True)
    
    df['Support_Strength_20D'] = df.groupby('Symbol').apply(
        lambda x: calculate_sr_strength(x['Close'], x['Support_20D'])
    ).reset_index(level=0, drop=True)
    
    # 7. Pivot points (classic)
    df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['Pivot_Point']) - df['Low']
    df['S1'] = (2 * df['Pivot_Point']) - df['High']
    df['R2'] = df['Pivot_Point'] + (df['High'] - df['Low'])
    df['S2'] = df['Pivot_Point'] - (df['High'] - df['Low'])
    
    # 8. Fibonacci retracement levels (based on recent high/low)
    recent_high = df.groupby('Symbol')['High'].transform(
        lambda x: x.rolling(window=50, min_periods=10).max()
    )
    recent_low = df.groupby('Symbol')['Low'].transform(
        lambda x: x.rolling(window=50, min_periods=10).min()
    )
    range_ = recent_high - recent_low
    
    df['Fib_23.6%'] = recent_low + (range_ * 0.236)
    df['Fib_38.2%'] = recent_low + (range_ * 0.382)
    df['Fib_50%'] = recent_low + (range_ * 0.5)
    df['Fib_61.8%'] = recent_low + (range_ * 0.618)
    df['Fib_78.6%'] = recent_low + (range_ * 0.786)
    
    # 9. Dynamic support/resistance (VWAP-based)
    df['Dynamic_Support'] = df['VWAP'] - (df['ATR_14'] * 1.5)
    df['Dynamic_Resistance'] = df['VWAP'] + (df['ATR_14'] * 1.5)
    
    # 10. Support/Resistance flip (when broken levels become new S/R)
    df['Previous_Resistance_Broken'] = (
        (df['Close'] > df['Resistance_20D']) & 
        (df['Close'].shift(1) <= df['Resistance_20D'].shift(1))
    ).astype(int)
    
    df['Previous_Support_Broken'] = (
        (df['Close'] < df['Support_20D']) & 
        (df['Close'].shift(1) >= df['Support_20D'].shift(1))
    ).astype(int)
    
    return df

# ==================================================
# MODULE 8: PERFORMANCE METRICS
# ==================================================

def calculate_performance_metrics(df):
    """
    MODULE 3 — PERFORMANCE & RETURNS
    Calculate comprehensive performance and return metrics
    """
    df = df.copy()
    
    # 1. Daily returns (already calculated)
    
    # 2. Cumulative returns (already calculated)
    
    # 3-4. Rolling returns (already calculated in basic features)
    
    # 5. Sharpe Ratio (risk-adjusted return)
    df['Sharpe_20D'] = df.groupby('Symbol')['Return'].transform(
        lambda x: (x.rolling(window=20, min_periods=10).mean() * np.sqrt(252)) / 
                  (x.rolling(window=20, min_periods=10).std() + 1e-10)
    )
    
    # 6. Sortino Ratio (downside risk-adjusted)
    def calculate_sortino(returns, window=20):
        """Calculate Sortino ratio focusing on downside deviation"""
        if len(returns) < window:
            return pd.Series(np.nan, index=returns.index)
        
        sortino = []
        for i in range(len(returns)):
            if i < window - 1:
                sortino.append(np.nan)
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                mean_return = window_returns.mean()
                downside_returns = window_returns[window_returns < 0]
                if len(downside_returns) > 0:
                    downside_dev = downside_returns.std()
                    sortino_ratio = (mean_return * np.sqrt(252)) / (downside_dev + 1e-10)
                else:
                    sortino_ratio = np.nan
                sortino.append(sortino_ratio)
        
        return pd.Series(sortino, index=returns.index)
    
    df['Sortino_20D'] = df.groupby('Symbol')['Return'].transform(
        lambda x: calculate_sortino(x, window=20)
    )
    
    # 7. Calmar Ratio (return vs max drawdown)
    df['Calmar_20D'] = (df['Return'].rolling(window=20).mean() * 252) / \
                       (-df['Max_Drawdown_20D'].replace(0, np.nan) + 1e-10)
    
    # 8. Information Ratio
    # First calculate market return (average across all symbols)
    market_return = df.groupby('Date')['Return'].mean()
    df = df.merge(market_return.rename('Market_Return'), on='Date', how='left')
    
    df['Excess_Return'] = df['Return'] - df['Market_Return']
    df['Information_Ratio_20D'] = df.groupby('Symbol')['Excess_Return'].transform(
        lambda x: (x.rolling(window=20, min_periods=10).mean() * np.sqrt(252)) /
                  (x.rolling(window=20, min_periods=10).std() + 1e-10)
    )
    
    # 9. Omega Ratio
    def calculate_omega(returns, threshold=0, window=20):
        """Calculate Omega ratio"""
        if len(returns) < window:
            return pd.Series(np.nan, index=returns.index)
        
        omega = []
        for i in range(len(returns)):
            if i < window - 1:
                omega.append(np.nan)
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                gains = window_returns[window_returns > threshold] - threshold
                losses = threshold - window_returns[window_returns < threshold]
                
                if len(gains) == 0:
                    omega.append(0)
                elif len(losses) == 0:
                    omega.append(np.inf)
                else:
                    omega_ratio = gains.sum() / (losses.sum() + 1e-10)
                    omega.append(omega_ratio)
        
        return pd.Series(omega, index=returns.index)
    
    df['Omega_Ratio_20D'] = df.groupby('Symbol')['Return'].transform(
        lambda x: calculate_omega(x, threshold=0, window=20)
    )
    
    # 10. Sterling Ratio
    df['Sterling_Ratio_20D'] = (df['Return'].rolling(window=20).mean() * 252) / \
                                (abs(df['Max_Drawdown_20D']) + 1e-10)
    
    # 11. Burke Ratio
    def calculate_burke(returns, window=20):
        """Calculate Burke ratio (alternative to Calmar)"""
        if len(returns) < window:
            return pd.Series(np.nan, index=returns.index)
        
        burke = []
        for i in range(len(returns)):
            if i < window - 1:
                burke.append(np.nan)
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                cumulative_return = (1 + window_returns).prod() - 1
                drawdowns = []
                peak = 1
                for ret in window_returns:
                    peak = max(peak, peak * (1 + ret))
                    drawdown = (peak * (1 + ret) - peak) / peak
                    drawdowns.append(drawdown)
                
                mdd_squared_sum = sum([dd**2 for dd in drawdowns if dd < 0])
                if mdd_squared_sum > 0:
                    burke_ratio = cumulative_return / np.sqrt(mdd_squared_sum)
                else:
                    burke_ratio = np.nan
                burke.append(burke_ratio)
        
        return pd.Series(burke, index=returns.index)
    
    df['Burke_Ratio_20D'] = df.groupby('Symbol')['Return'].transform(
        lambda x: calculate_burke(x, window=20)
    )
    
    # 12. Annualized return
    df['Annualized_Return'] = df.groupby('Symbol')['Return'].transform(
        lambda x: x.mean() * 252
    )
    
    # 13. Annualized volatility
    df['Annualized_Volatility'] = df.groupby('Symbol')['Return'].transform(
        lambda x: x.std() * np.sqrt(252) * 100
    )
    
    # 14. Downside deviation
    df['Downside_Deviation_20D'] = df.groupby('Symbol').apply(
        lambda x: x['Return'].rolling(window=20).apply(
            lambda y: y[y < 0].std() if len(y[y < 0]) > 0 else 0
        )
    ).reset_index(level=0, drop=True) * np.sqrt(252) * 100
    
    # 15. Upside/downside capture ratios
    df['Upside_Capture'] = df.groupby('Symbol').apply(
        lambda x: x[x['Market_Return'] > 0]['Return'].mean() / 
                  x[x['Market_Return'] > 0]['Market_Return'].mean()
    ).reset_index(level=0, drop=True)
    
    df['Downside_Capture'] = df.groupby('Symbol').apply(
        lambda x: x[x['Market_Return'] < 0]['Return'].mean() / 
                  x[x['Market_Return'] < 0]['Market_Return'].mean()
    ).reset_index(level=0, drop=True)
    
    return df

# ==================================================
# MODULE 9: PEER & RELATIVE ANALYSIS
# ==================================================

def calculate_peer_metrics(df):
    """
    MODULE 6 — PEER & RELATIVE ANALYSIS
    Calculate peer comparison and relative performance metrics
    """
    df = df.copy()
    
    # 1. Alpha (excess return over market)
    df['Alpha'] = df['Return'] - df['Market_Return']
    
    # 2. Cumulative Alpha
    df['Cumulative_Alpha'] = df.groupby('Symbol')['Alpha'].apply(
        lambda x: (1 + x.fillna(0)).cumprod() - 1
    ).reset_index(drop=True) * 100
    
    # 3. Relative performance vs peer average
    peer_avg_return = df.groupby(['Date', 'Symbol'])['Return'].transform('mean')
    df['Relative_Performance'] = df['Return'] - peer_avg_return
    
    # 4. Peer volatility comparison
    peer_avg_vol = df.groupby('Date')['Volatility_20D'].mean()
    df = df.merge(peer_avg_vol.rename('Peer_Avg_Volatility'), on='Date', how='left')
    df['Relative_Volatility'] = df['Volatility_20D'] / df['Peer_Avg_Volatility']
    
    # 5. Relative strength ranking
    df['Return_Rank'] = df.groupby('Date')['Return'].rank(pct=True)
    df['Volatility_Rank'] = df.groupby('Date')['Volatility_20D'].rank(pct=True, ascending=False)
    df['Sharpe_Rank'] = df.groupby('Date')['Sharpe_20D'].rank(pct=True)
    
    # 6. Outperformance frequency
    df['Outperforms_Market'] = (df['Return'] > df['Market_Return']).astype(int)
    df['Outperformance_Streak'] = df.groupby('Symbol')['Outperforms_Market'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # 7. Peer dominance score
    df['Peer_Dominance_Score'] = (
        df['Return_Rank'] * 0.4 +
        df['Sharpe_Rank'] * 0.3 +
        (1 - df['Volatility_Rank']) * 0.3
    )
    
    # 8. Beta (systematic risk)
    def calculate_beta(returns, market_returns, window=60):
        """Calculate rolling beta"""
        if len(returns) < window:
            return pd.Series(np.nan, index=returns.index)
        
        beta = []
        for i in range(len(returns)):
            if i < window - 1:
                beta.append(np.nan)
            else:
                stock_returns = returns.iloc[i-window+1:i+1]
                market_ret = market_returns.iloc[i-window+1:i+1]
                cov = np.cov(stock_returns, market_ret)[0, 1]
                var = np.var(market_ret)
                if var > 0:
                    beta.append(cov / var)
                else:
                    beta.append(np.nan)
        
        return pd.Series(beta, index=returns.index)
    
    df['Beta_60D'] = df.groupby('Symbol').apply(
        lambda x: calculate_beta(x['Return'], x['Market_Return'], window=60)
    ).reset_index(level=0, drop=True)
    
    # 9. Correlation with market
    df['Correlation_60D'] = df.groupby('Symbol').apply(
        lambda x: x['Return'].rolling(window=60).corr(x['Market_Return'])
    ).reset_index(level=0, drop=True)
    
    # 10. Relative Strength Index (peer-based)
    df['RSI_Peer'] = df.groupby('Date').apply(
        lambda x: (x['Return'].rank(pct=True) - 0.5) * 2
    ).reset_index(level=0, drop=True)
    
    return df

# ==================================================
# MODULE 10: MARKET STRUCTURE & REGIMES
# ==================================================

def calculate_market_structure(df):
    """
    MODULE 7 — MARKET STRUCTURE & REGIMES
    Calculate market structure, regimes, and phase classification
    """
    df = df.copy()
    
    # 1. Daily range expansion/contraction
    df['Range_Expansion'] = df.groupby('Symbol')['Intraday_Range'].transform(
        lambda x: x / x.rolling(window=20, min_periods=5).mean()
    )
    
    # 2. Momentum regime map
    def classify_momentum_regime(returns, window=20):
        """Classify momentum regime based on rolling returns"""
        regimes = []
        for i in range(len(returns)):
            if i < window - 1:
                regimes.append('Undefined')
            else:
                rolling_return = ((1 + returns.iloc[i-window+1:i+1]).prod() - 1) * 100
                if rolling_return > 10:
                    regimes.append('Strong Bullish')
                elif rolling_return > 5:
                    regimes.append('Bullish')
                elif rolling_return > -5:
                    regimes.append('Neutral')
                elif rolling_return > -10:
                    regimes.append('Bearish')
                else:
                    regimes.append('Strong Bearish')
        return pd.Series(regimes, index=returns.index)
    
    df['Momentum_Regime'] = df.groupby('Symbol')['Return'].transform(
        lambda x: classify_momentum_regime(x, window=20)
    )
    
    # 3. Market phase classification
    def classify_market_phase(trend_slope, volatility, window=20):
        """Classify market phase: Accumulation, Markup, Distribution, Decline"""
        phases = []
        for i in range(len(trend_slope)):
            if pd.isna(trend_slope.iloc[i]) or pd.isna(volatility.iloc[i]):
                phases.append('Undefined')
            else:
                slope = trend_slope.iloc[i]
                vol = volatility.iloc[i]
                
                if slope > 0 and vol < 25:
                    phases.append('Accumulation')
                elif slope > 0 and vol >= 25:
                    phases.append('Markup')
                elif slope < 0 and vol < 25:
                    phases.append('Distribution')
                elif slope < 0 and vol >= 25:
                    phases.append('Decline')
                else:
                    phases.append('Consolidation')
        return pd.Series(phases, index=trend_slope.index)
    
    df['Market_Phase'] = classify_market_phase(df['Trend_Slope_20D'], df['Volatility_20D'])
    
    # 4. Trend exhaustion signal
    def detect_trend_exhaustion(close, volume, window=20):
        """Detect potential trend exhaustion using volume divergence"""
        exhaustion = pd.Series(0, index=close.index)
        for i in range(window, len(close)):
            price_trend = close.iloc[i] - close.iloc[i-window]
            volume_trend = volume.iloc[i] - volume.iloc[i-window]
            
            # Bullish trend exhaustion: price up, volume down
            if price_trend > 0 and volume_trend < 0:
                exhaustion.iloc[i] = 1
            # Bearish trend exhaustion: price down, volume down
            elif price_trend < 0 and volume_trend < 0:
                exhaustion.iloc[i] = -1
        
        return exhaustion
    
    df['Trend_Exhaustion'] = df.groupby('Symbol').apply(
        lambda x: detect_trend_exhaustion(x['Close'], x['Volume'])
    ).reset_index(level=0, drop=True)
    
    # 5. Consolidation vs expansion map
    df['Consolidation_Expansion_Ratio'] = df['BB_Width'] / df['BB_Width'].rolling(50).mean()
    df['Market_State'] = np.where(
        df['Consolidation_Expansion_Ratio'] < 0.8, 'Consolidation',
        np.where(df['Consolidation_Expansion_Ratio'] > 1.2, 'Expansion', 'Neutral')
    )
    
    # 6. Market breadth (for multi-symbol analysis)
    if len(df['Symbol'].unique()) > 1:
        df['Advancing_Issues'] = df.groupby('Date')['Return'].transform(
            lambda x: (x > 0).sum()
        )
        df['Declining_Issues'] = df.groupby('Date')['Return'].transform(
            lambda x: (x < 0).sum()
        )
        df['Advance_Decline_Line'] = (df['Advancing_Issues'] - df['Declining_Issues']).cumsum()
    
    # 7. Market volatility regime
    df['Market_Volatility_Regime'] = pd.cut(
        df['Volatility_20D'],
        bins=[-np.inf, 15, 30, 50, np.inf],
        labels=['Low Volatility', 'Medium Volatility', 'High Volatility', 'Extreme Volatility']
    )
    
    # 8. Price efficiency ratio
    df['Price_Efficiency_Ratio'] = abs(df['Return']) / (df['Range_Pct'] + 1e-10)
    
    # 9. Trend quality score
    df['Trend_Quality_Score'] = (
        np.tanh(abs(df['Trend_Slope_20D']) * 10) * 0.4 +
        (df['Trend_Persistence_20D'].abs() * 0.3) +
        (1 - df['Price_Efficiency_Ratio'].rolling(20).std() / 
         (df['Price_Efficiency_Ratio'].rolling(20).std() + 1e-10)) * 0.3
    )
    
    # 10. Market regime composite
    def calculate_market_regime_composite(df_row):
        """Composite market regime indicator"""
        score = 0
        
        # Trend component
        if df_row['Trend_Slope_20D'] > 1:
            score += 1
        elif df_row['Trend_Slope_20D'] < -1:
            score -= 1
        
        # Momentum component
        if df_row['Return'] > 0.01:
            score += 0.5
        elif df_row['Return'] < -0.01:
            score -= 0.5
        
        # Volatility component
        if df_row['Volatility_20D'] > 30:
            score -= 0.5
        
        return score
    
    df['Market_Regime_Composite'] = df.apply(calculate_market_regime_composite, axis=1)
    
    return df

# ==================================================
# MODULE 11: ML MODEL INTEGRATION
# ==================================================

def load_ml_models():
    """
    Load pre-trained ML models for direction and return prediction
    """
    try:
        print("🔄 Loading ML models...")
        
        # Load models
        with open(MODEL_CONFIG['direction_model'], 'rb') as f:
            direction_model = pickle.load(f)
        
        with open(MODEL_CONFIG['return_model'], 'rb') as f:
            return_model = pickle.load(f)
        
        # Load scaler
        with open(MODEL_CONFIG['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        
        # Load symbol encoder
        with open(MODEL_CONFIG['symbol_encoder'], 'rb') as f:
            symbol_encoder = pickle.load(f)
        
        print("✅ ML models loaded successfully")
        return direction_model, return_model, scaler, symbol_encoder
    
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML models: {str(e)}")
        print("   ML predictions will not be available")
        return None, None, None, None

def prepare_features_for_prediction(df, scaler, symbol_encoder):
    """
    Prepare features for ML model prediction
    """
    df = df.copy()
    
    # Features used by the model (adjust based on your actual model)
    feature_columns = [
        'Return', 'Volatility_20D', 'RSI_14', 'MACD', 'MACD_Histogram',
        'BB_Width', 'ATR_Pct', 'VWAP_Deviation_Pct', 'Trend_Slope_20D',
        'Sharpe_20D', 'Momentum_20D', 'Dist_MA20_Pct', 'Dist_MA50_Pct',
        'Volume_Ratio', 'Price_Acceleration'
    ]
    
    # Ensure all features exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Create feature matrix
    X = df[feature_columns].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    return X_scaled, feature_columns

def predict_with_ml_models(df, direction_model, return_model, scaler, symbol_encoder):
    """
    Make predictions using loaded ML models
    """
    if direction_model is None or return_model is None:
        print("⚠️ ML models not loaded, skipping predictions")
        return df
    
    try:
        df = df.copy()
        
        # Prepare features
        X_scaled, feature_cols = prepare_features_for_prediction(df, scaler, symbol_encoder)
        
        # Direction prediction (classification: 1 = up, 0 = down)
        direction_pred = direction_model.predict(X_scaled)
        direction_proba = direction_model.predict_proba(X_scaled)[:, 1]  # Probability of up move
        
        df['ML_Direction_Prediction'] = direction_pred
        df['ML_Direction_Probability'] = direction_proba
        
        # Return prediction (regression: predicted return %)
        return_pred = return_model.predict(X_scaled)
        df['ML_Return_Prediction'] = return_pred * 100  # Convert to percentage
        
        # Confidence score
        df['ML_Confidence_Score'] = np.abs(direction_proba - 0.5) * 2  # Scale to [0, 1]
        
        # ML-based signal
        df['ML_Signal'] = np.where(
            (df['ML_Direction_Prediction'] == 1) & (df['ML_Confidence_Score'] > 0.6),
            'BUY',
            np.where(
                (df['ML_Direction_Prediction'] == 0) & (df['ML_Confidence_Score'] > 0.6),
                'SELL',
                'HOLD'
            )
        )
        
        print(f"✅ ML predictions generated for {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"⚠️ Error during ML prediction: {str(e)}")
        return df

# ==================================================
# MODULE 12: DECISION ENGINE
# ==================================================

def calculate_decision_score(row):
    """
    MODULE 9 — DECISION ENGINE & EXECUTION
    Calculate comprehensive decision score based on multiple factors
    """
    score = 0
    
    # 1. Trend factors (30% weight)
    trend_weight = 0.30
    
    if pd.notna(row.get('Trend_Slope_20D')):
        slope = row['Trend_Slope_20D']
        if slope > 2:
            score += trend_weight * 0.8
        elif slope > 1:
            score += trend_weight * 0.5
        elif slope > 0.5:
            score += trend_weight * 0.3
        elif slope < -2:
            score += trend_weight * -0.8
        elif slope < -1:
            score += trend_weight * -0.5
        elif slope < -0.5:
            score += trend_weight * -0.3
    
    if pd.notna(row.get('MA20_Above_MA50')):
        score += trend_weight * 0.2 * (row['MA20_Above_MA50'] * 2 - 1)
    
    # 2. Momentum factors (25% weight)
    momentum_weight = 0.25
    
    if pd.notna(row.get('RSI_14')):
        rsi = row['RSI_14']
        if 40 <= rsi <= 60:
            score += momentum_weight * 0.2  # Neutral
        elif 30 <= rsi < 40:
            score += momentum_weight * 0.5  # Oversold
        elif rsi < 30:
            score += momentum_weight * 0.8  # Extremely oversold
        elif 60 < rsi <= 70:
            score += momentum_weight * -0.3  # Overbought
        elif rsi > 70:
            score += momentum_weight * -0.6  # Extremely overbought
    
    if pd.notna(row.get('MACD_Histogram')):
        macd_hist = row['MACD_Histogram']
        if macd_hist > 0:
            score += momentum_weight * 0.3
        else:
            score += momentum_weight * -0.3
    
    # 3. Risk factors (20% weight)
    risk_weight = 0.20
    
    if pd.notna(row.get('Volatility_20D')):
        vol = row['Volatility_20D']
        if vol < 15:
            score += risk_weight * 0.5  # Low risk
        elif vol > 40:
            score += risk_weight * -0.5  # High risk
    
    if pd.notna(row.get('Max_Drawdown_20D')):
        mdd = row['Max_Drawdown_20D']
        if mdd > -5:
            score += risk_weight * 0.4
        elif mdd < -15:
            score += risk_weight * -0.4
    
    # 4. Value factors (15% weight)
    value_weight = 0.15
    
    if pd.notna(row.get('Dist_MA50_Pct')):
        dist = row['Dist_MA50_Pct']
        if dist < -10:
            score += value_weight * 0.6  # Undervalued
        elif dist > 10:
            score += value_weight * -0.3  # Overvalued
    
    if pd.notna(row.get('VWAP_Deviation_Pct')):
        vwap_dev = row['VWAP_Deviation_Pct']
        if vwap_dev < -2:
            score += value_weight * 0.4  # Below VWAP
        elif vwap_dev > 2:
            score += value_weight * -0.2  # Above VWAP
    
    # 5. ML prediction factors (10% weight)
    ml_weight = 0.10
    
    if pd.notna(row.get('ML_Direction_Probability')):
        ml_prob = row['ML_Direction_Probability']
        ml_conf = row.get('ML_Confidence_Score', 0)
        score += ml_weight * (ml_prob * 2 - 1) * ml_conf
    
    # Normalize score to [-1, 1]
    score = max(min(score, 1), -1)
    
    return score

def get_decision_label(score):
    """
    Convert decision score to label
    """
    if score >= DECISION_THRESHOLDS['strong_buy']:
        return "🚀 STRONG BUY"
    elif score >= DECISION_THRESHOLDS['buy']:
        return "✅ BUY"
    elif score >= DECISION_THRESHOLDS['accumulate']:
        return "📈 ACCUMULATE"
    elif score >= DECISION_THRESHOLDS['hold']:
        return "⏸️ HOLD"
    elif score >= DECISION_THRESHOLDS['reduce']:
        return "⚠️ REDUCE"
    elif score >= DECISION_THRESHOLDS['sell']:
        return "❌ SELL"
    else:
        return "⚠️ STRONG SELL"

def calculate_confidence_score(row):
    """
    Calculate confidence in the decision
    """
    confidence = 0
    
    # Signal strength
    if pd.notna(row.get('Trend_Slope_20D')):
        confidence += min(abs(row['Trend_Slope_20D']) / 5, 0.3)
    
    # Volume confirmation
    if pd.notna(row.get('Volume_Ratio')):
        if row['Volume_Ratio'] > 1.2:
            confidence += 0.2
    
    # Multiple timeframe confirmation
    ma_confirmation = 0
    if pd.notna(row.get('Dist_MA20_Pct')) and pd.notna(row.get('Dist_MA50_Pct')):
        if (row['Dist_MA20_Pct'] > 0 and row['Dist_MA50_Pct'] > 0):
            ma_confirmation += 0.3
        elif (row['Dist_MA20_Pct'] < 0 and row['Dist_MA50_Pct'] < 0):
            ma_confirmation += 0.3
    confidence += ma_confirmation
    
    # RSI confirmation
    if pd.notna(row.get('RSI_14')):
        rsi = row['RSI_14']
        if 35 <= rsi <= 65:
            confidence += 0.2
    
    # ML confidence
    if pd.notna(row.get('ML_Confidence_Score')):
        confidence += row['ML_Confidence_Score'] * 0.2
    
    return min(confidence, 1.0)

def calculate_risk_metrics(row):
    """
    Calculate comprehensive risk metrics
    """
    risk_score = 0
    risk_flags = []
    
    # Volatility risk
    if pd.notna(row.get('Volatility_20D')):
        vol = row['Volatility_20D']
        if vol > 40:
            risk_score += 3
            risk_flags.append('⚠️ High Volatility')
        elif vol > 30:
            risk_score += 2
            risk_flags.append('⚠️ Moderate Volatility')
    
    # Drawdown risk
    if pd.notna(row.get('Max_Drawdown_20D')):
        mdd = row['Max_Drawdown_20D']
        if mdd < -20:
            risk_score += 4
            risk_flags.append('🚨 Severe Drawdown')
        elif mdd < -10:
            risk_score += 2
            risk_flags.append('⚠️ Significant Drawdown')
    
    # Overbought/Oversold risk
    if pd.notna(row.get('RSI_14')):
        rsi = row['RSI_14']
        if rsi > 80:
            risk_score += 2
            risk_flags.append('⚠️ Overbought')
        elif rsi < 20:
            risk_score += 1
            risk_flags.append('ℹ️ Oversold')
    
    # Trend risk
    if pd.notna(row.get('Trend_Slope_20D')):
        if row['Trend_Slope_20D'] < -3:
            risk_score += 2
            risk_flags.append('⚠️ Strong Downtrend')
    
    # Determine risk level
    if risk_score >= 5:
        risk_level = 'HIGH'
    elif risk_score >= 3:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    if not risk_flags:
        risk_flags.append('✅ Low Risk Profile')
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_flags': risk_flags
    }

# ==================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ==================================================

def engineer_features(df):
    """
    Main feature engineering pipeline
    Applies all feature engineering modules in sequence
    """
    if df.empty:
        return df
    
    print("🔧 Starting comprehensive feature engineering...")
    
    # Ensure required columns exist
    if 'Prev Close' not in df.columns:
        df['Prev Close'] = df.groupby('Symbol')['Close'].shift(1)
    if 'Volume' not in df.columns:
        df['Volume'] = 1000000
    
    # Apply feature engineering modules in sequence
    print("  → Basic price features...")
    df = calculate_basic_price_features(df)
    
    print("  → Moving averages...")
    df = calculate_moving_averages(df)
    
    print("  → Volatility metrics...")
    df = calculate_volatility_metrics(df)
    
    print("  → VWAP features...")
    df = calculate_vwap_features(df)
    
    print("  → Technical indicators...")
    df = calculate_technical_indicators(df)
    
    print("  → Support & resistance...")
    df = calculate_support_resistance(df)
    
    print("  → Performance metrics...")
    df = calculate_performance_metrics(df)
    
    print("  → Peer metrics...")
    df = calculate_peer_metrics(df)
    
    print("  → Market structure...")
    df = calculate_market_structure(df)
    
    print(f"✅ Feature engineering complete: {len(df.columns)} features generated")
    return df

def prepare_dashboard_data(df, selected_symbols):
    """
    Prepare summary data for dashboard display
    """
    if df.empty or not selected_symbols:
        return pd.DataFrame()
    
    try:
        # Get latest data for selected symbols
        latest_data = df[df['Symbol'].isin(selected_symbols)].groupby('Symbol').last().reset_index()
        
        # Calculate decision metrics
        latest_data['Decision_Score'] = latest_data.apply(calculate_decision_score, axis=1)
        latest_data['Decision_Label'] = latest_data['Decision_Score'].apply(get_decision_label)
        latest_data['Confidence_Score'] = latest_data.apply(calculate_confidence_score, axis=1)
        
        # Calculate risk metrics
        risk_metrics = latest_data.apply(calculate_risk_metrics, axis=1, result_type='expand')
        latest_data = pd.concat([latest_data, risk_metrics], axis=1)
        
        # Rename risk_level column
        if 'risk_level' in latest_data.columns:
            latest_data = latest_data.rename(columns={'risk_level': 'Risk_Level'})
        
        return latest_data
        
    except Exception as e:
        print(f"Error preparing dashboard  {e}")
        return pd.DataFrame()

# ==================================================
# SAMPLE DATA GENERATION
# ==================================================

def create_sample_data():
    """
    Create realistic sample NIFTY-50 data for demo purposes
    """
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='B')
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
               'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'LT',
               'HINDUNILVR', 'BAJFINANCE', 'AXISBANK', 'WIPRO', 'TECHM']
    data = []
    
    for symbol in symbols:
        base_price = np.random.uniform(1500, 3500)
        trend_factor = np.random.uniform(0.8, 1.2)
        volatility_factor = np.random.uniform(0.01, 0.03)
        
        for i, date in enumerate(dates):
            # Simulate realistic price movement with trend + seasonality + noise
            trend = (i / len(dates)) * trend_factor
            seasonality = np.sin(i / 30) * 0.3 + np.sin(i / 100) * 0.2
            noise = np.random.normal(0, volatility_factor)
            
            close = base_price * (1 + trend + seasonality + noise)
            
            # Generate OHLC with realistic relationships
            high = close * (1 + abs(np.random.normal(0, 0.015)))
            low = close * (1 - abs(np.random.normal(0, 0.015)))
            low = min(low, close * 0.98)
            high = max(high, close * 1.02)
            open_price = np.random.uniform(low, high)
            prev_close = close * (1 + np.random.normal(0, 0.01)) if i > 0 else open_price
            
            # Calculate VWAP
            vwap = (high + low + close * 2) / 4
            
            # Volume with some correlation to price movement
            volume_base = np.random.randint(500000, 5000000)
            volume = volume_base * (1 + abs(noise) * 10)
            
            data.append({
                'Date': date,
                'Symbol': symbol,
                'Series': 'EQ',
                'Prev Close': round(prev_close, 2),
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Last': round(close, 2),
                'Close': round(close, 2),
                'VWAP': round(vwap, 2),
                'Volume': int(volume)
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated sample data: {len(df)} rows, {len(df['Symbol'].unique())} symbols")
    return df

# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def get_market_overview(df):
    """
    Generate market overview statistics
    """
    if df.empty:
        return {}
    
    overview = {}
    
    # Latest date
    overview['latest_date'] = df['Date'].max().strftime('%Y-%m-%d')
    overview['total_symbols'] = len(df['Symbol'].unique())
    overview['total_records'] = len(df)
    
    # Market breadth
    if 'Return' in df.columns:
        latest_returns = df.groupby('Symbol').last()['Return']
        overview['advancing'] = (latest_returns > 0).sum()
        overview['declining'] = (latest_returns < 0).sum()
        overview['unchanged'] = (latest_returns == 0).sum()
    
    # Average metrics
    if 'Return' in df.columns:
        overview['avg_daily_return'] = df.groupby('Symbol')['Return'].mean().mean() * 100
    if 'Volatility_20D' in df.columns:
        overview['avg_volatility'] = df.groupby('Symbol')['Volatility_20D'].mean().mean()
    if 'Sharpe_20D' in df.columns:
        overview['avg_sharpe'] = df.groupby('Symbol')['Sharpe_20D'].mean().mean()
    
    # Market sentiment
    if 'RSI_14' in df.columns:
        latest_rsi = df.groupby('Symbol').last()['RSI_14']
        overview['overbought'] = (latest_rsi > 70).sum()
        overview['oversold'] = (latest_rsi < 30).sum()
        overview['neutral'] = ((latest_rsi >= 30) & (latest_rsi <= 70)).sum()
    
    return overview

def filter_data_by_date_range(df, start_date, end_date):
    """
    Filter dataframe by date range
    """
    return df[
        (df['Date'] >= pd.to_datetime(start_date)) & 
        (df['Date'] <= pd.to_datetime(end_date))
    ].copy()

def get_stock_metrics(df, symbol):
    """
    Get all metrics for a specific stock
    """
    stock_data = df[df['Symbol'] == symbol].copy()
    if stock_data.empty:
        return None
    
    latest = stock_data.iloc[-1]
    
    metrics = {
        'current_price': latest['Close'],
        'daily_return': latest['Return'] * 100,
        'volatility': latest['Volatility_20D'],
        'rsi': latest['RSI_14'],
        'macd': latest['MACD'],
        'sharpe': latest['Sharpe_20D'],
        'max_drawdown': latest['Max_Drawdown_20D'],
        'vwap_deviation': latest['VWAP_Deviation_Pct'],
        'trend_slope': latest['Trend_Slope_20D'],
        'decision_score': latest['Decision_Score'],
        'decision_label': latest['Decision_Label'],
        'confidence': latest['Confidence_Score'],
        'risk_level': latest['Risk_Level']
    }
    
    return metrics

# ==================================================
# EXPORT FUNCTIONS
# ==================================================

def export_to_csv(df, filename='stock_analysis.csv'):
    """
    Export dataframe to CSV
    """
    df.to_csv(filename, index=False)
    print(f"✅ Data exported to {filename}")

def export_recommendations(recommendations_df, filename='recommendations.csv'):
    """
    Export recommendations to CSV
    """
    cols_to_export = ['Symbol', 'Decision_Label', 'Decision_Score', 'Confidence_Score',
                     'Risk_Level', 'Close', 'Return_Pct', 'Volatility_20D', 'RSI_14',
                     'Sharpe_20D', 'Max_Drawdown_20D']
    available_cols = [col for col in cols_to_export if col in recommendations_df.columns]
    
    recommendations_df[available_cols].to_csv(filename, index=False)
    print(f"✅ Recommendations exported to {filename}")

# ==================================================
# END OF UTILS.PY
# ==================================================