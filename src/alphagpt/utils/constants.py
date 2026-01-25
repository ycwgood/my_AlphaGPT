"""Project constants definition"""

# Backtest related constants
WARMUP_PERIOD = 20  # Warmup period for stabilizing factor values
POSITION_SCALE = 0.3  # Position scaling coefficient

# Factor calculation related constants
MIN_SAMPLE_SIZE = 20  # Minimum sample size for calculating Sharpe ratio
EPSILON = 1e-6  # Small constant to prevent division by zero
FACTOR_CLIP_MIN = -10  # Factor value lower bound for outlier clipping
FACTOR_CLIP_MAX = 10   # Factor value upper bound for outlier clipping

# Data processing related constants
RETURN_CLIP_LOWER = -0.1  # Return lower bound
RETURN_CLIP_UPPER = 0.1   # Return upper bound
RETURN_5D_CLIP_LOWER = -0.15  # 5-day return lower bound
RETURN_5D_CLIP_UPPER = 0.15   # 5-day return upper bound

# Feature engineering related constants
ROLLING_WINDOW = 40  # Rolling window size for feature normalization
MIN_PERIODS = 10  # Minimum observation period
FEATURE_CLIP_LOWER = -3  # Feature normalization lower bound
FEATURE_CLIP_UPPER = 3   # Feature normalization upper bound
VOL_CHG_CLIP_LOWER = -3  # Volume change lower bound
VOL_CHG_CLIP_UPPER = 3   # Volume change upper bound
TREND_CLIP_LOWER = -0.5  # Trend indicator lower bound
TREND_CLIP_UPPER = 0.5   # Trend indicator upper bound

# Data validation related constants
MIN_TRAIN_SAMPLES = 100  # Minimum training sample size
MIN_TEST_SAMPLES = 20    # Minimum test sample size

# Technical indicator periods
MA3_PERIOD = 3    # 3-day moving average period
MA5_PERIOD = 5    # 5-day moving average period
MA10_PERIOD = 10  # 10-day moving average period
MA15_PERIOD = 15  # 15-day moving average period
MA20_PERIOD = 20  # 20-day moving average period
MA30_PERIOD = 30  # 30-day moving average period
MA40_PERIOD = 40  # 40-day moving average period
MA60_PERIOD = 60  # 60-day moving average period
MA120_PERIOD = 120  # 120-day moving average period
MOM3_PERIOD = 3   # 3-day momentum period
MOM5_PERIOD = 5   # 5-day momentum period
MOM10_PERIOD = 10 # 10-day momentum period
MOM15_PERIOD = 15 # 15-day momentum period
MOM20_PERIOD = 20 # 20-day momentum period
RET3_PERIOD = 3   # 3-day return period
RET5_PERIOD = 5   # 5-day return period
RET10_PERIOD = 10 # 10-day return period
RET15_PERIOD = 15 # 15-day return period
RET20_PERIOD = 20 # 20-day return period
RET30_PERIOD = 30 # 30-day return period
RET60_PERIOD = 60 # 60-day return period
RSI_PERIOD = 14   # RSI calculation period
RSI_SHORT_PERIOD = 6   # Short RSI period
RSI_LONG_PERIOD = 21   # Long RSI period
ATR_PERIOD = 14   # ATR calculation period
ATR_SHORT_PERIOD = 7   # Short ATR period
ATR_LONG_PERIOD = 21   # Long ATR period
STD_PERIOD = 20   # Standard deviation period
STD_SHORT_PERIOD = 10  # Short std period
STD_LONG_PERIOD = 40   # Long std period
EMA_PERIOD = 12   # EMA period
EMA_LONG_PERIOD = 26  # Long EMA period
MACD_SIGNAL_PERIOD = 9  # MACD signal period
BOLLINGER_PERIOD = 20  # Bollinger Bands period
STOCH_K_PERIOD = 14  # Stochastic K period
STOCH_D_PERIOD = 3   # Stochastic D period

# Feature clipping bounds
VOLATILITY_CLIP_LOWER = -0.5  # Volatility lower bound
VOLATILITY_CLIP_UPPER = 0.5   # Volatility upper bound
HIGHLOW_CLIP_LOWER = -0.3     # High-low range lower bound
HIGHLOW_CLIP_UPPER = 0.3      # High-low range upper bound
GAP_CLIP_LOWER = -0.1         # Gap lower bound
GAP_CLIP_UPPER = 0.1          # Gap upper bound
RSI_CLIP_LOWER = -5           # RSI feature lower bound
RSI_CLIP_UPPER = 5            # RSI feature upper bound
MA_RATIO_CLIP_LOWER = -0.5    # MA ratio lower bound (expanded)
MA_RATIO_CLIP_UPPER = 0.5     # MA ratio upper bound (expanded)
BOLLINGER_CLIP_LOWER = -5     # Bollinger band position lower bound
BOLLINGER_CLIP_UPPER = 5      # Bollinger band position upper bound
STOCH_CLIP_LOWER = -5         # Stochastic lower bound
STOCH_CLIP_UPPER = 5          # Stochastic upper bound
MACD_CLIP_LOWER = -0.1        # MACD lower bound
MACD_CLIP_UPPER = 0.1         # MACD upper bound
SKEW_CLIP_LOWER = -5          # Skewness lower bound
SKEW_CLIP_UPPER = 5           # Skewness upper bound
KURT_CLIP_LOWER = -5          # Kurtosis lower bound
KURT_CLIP_UPPER = 5           # Kurtosis upper bound

# Annualization factor
TRADING_DAYS_PER_YEAR = 252  # Number of trading days per year

# Output directory
OUTPUT_DIR = "results"  # Default output directory for results
