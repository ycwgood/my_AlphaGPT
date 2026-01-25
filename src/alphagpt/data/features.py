"""Feature engineering module - 100+ technical features"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from alphagpt.utils.validators import validate_dataframe, validate_price_data
from alphagpt.utils.constants import (
    RETURN_CLIP_LOWER, RETURN_CLIP_UPPER,
    RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER,
    VOL_CHG_CLIP_LOWER, VOL_CHG_CLIP_UPPER,
    TREND_CLIP_LOWER, TREND_CLIP_UPPER,
    FEATURE_CLIP_LOWER, FEATURE_CLIP_UPPER,
    ROLLING_WINDOW, MIN_PERIODS, EPSILON,
    MA3_PERIOD, MA5_PERIOD, MA10_PERIOD, MA15_PERIOD, MA20_PERIOD,
    MA30_PERIOD, MA40_PERIOD, MA60_PERIOD, MA120_PERIOD,
    MOM3_PERIOD, MOM5_PERIOD, MOM10_PERIOD, MOM15_PERIOD, MOM20_PERIOD,
    RET3_PERIOD, RET5_PERIOD, RET10_PERIOD, RET15_PERIOD, RET20_PERIOD,
    RET30_PERIOD, RET60_PERIOD,
    RSI_PERIOD, RSI_SHORT_PERIOD, RSI_LONG_PERIOD,
    ATR_PERIOD, ATR_SHORT_PERIOD, ATR_LONG_PERIOD,
    STD_PERIOD, STD_SHORT_PERIOD, STD_LONG_PERIOD,
    EMA_PERIOD, EMA_LONG_PERIOD, MACD_SIGNAL_PERIOD,
    BOLLINGER_PERIOD, STOCH_K_PERIOD, STOCH_D_PERIOD,
    VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER,
    HIGHLOW_CLIP_LOWER, HIGHLOW_CLIP_UPPER,
    GAP_CLIP_LOWER, GAP_CLIP_UPPER,
    RSI_CLIP_LOWER, RSI_CLIP_UPPER,
    MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER,
    BOLLINGER_CLIP_LOWER, BOLLINGER_CLIP_UPPER,
    STOCH_CLIP_LOWER, STOCH_CLIP_UPPER,
    MACD_CLIP_LOWER, MACD_CLIP_UPPER,
    SKEW_CLIP_LOWER, SKEW_CLIP_UPPER,
    KURT_CLIP_LOWER, KURT_CLIP_UPPER,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create and transform 100+ technical features for factor generation"""

    @staticmethod
    def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + EPSILON)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_stochastic(
        df: pd.DataFrame,
        k_period: int = STOCH_K_PERIOD,
        d_period: int = STOCH_D_PERIOD
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']

        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + EPSILON)
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()

        return k_percent, d_percent

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        prev_close = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    @staticmethod
    def _calculate_bollinger_bands(
        series: pd.Series,
        period: int = BOLLINGER_PERIOD,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def _calculate_skewness(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling skewness"""
        return series.rolling(window=window, min_periods=1).skew()

    @staticmethod
    def _calculate_kurtosis(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling kurtosis"""
        return series.rolling(window=window, min_periods=1).kurt()

    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']

        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()

        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low + EPSILON)
        return williams_r

    @staticmethod
    def _calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']

        typical_price = (high + low + df['close']) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
        mad = typical_price.rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
        )

        cci = (typical_price - sma_tp) / (0.015 * mad + EPSILON)
        return cci

    @staticmethod
    def _calculate_money_flow_index(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Calculate Money Flow Index"""
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']

        typical_price = (high + low + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Calculate positive and negative money flow
        flow_sign = np.where(typical_price.diff() > 0, 1, -1)
        positive_flow = (money_flow * flow_sign).clip(lower=0)
        negative_flow = (-money_flow * flow_sign).clip(lower=0)

        positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + EPSILON)))
        return mfi

    @staticmethod
    def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create 100+ technical features from price data

        Args:
            df: DataFrame with price data (requires close, volume; optionally high, low)

        Returns:
            Tuple of (DataFrame with features, list of feature names)
        """
        # Validate input data
        validate_dataframe(df, ['close', 'volume'])
        validate_price_data(df)

        df = df.copy()
        has_highlow = all(col in df.columns for col in ['high', 'low'])

        # ============================================================
        # SECTION 1: RETURN FEATURES (12 features)
        # ============================================================
        df['ret'] = df['close'].pct_change().fillna(0).clip(RETURN_CLIP_LOWER, RETURN_CLIP_UPPER)
        df['ret3'] = df['close'].pct_change(RET3_PERIOD).fillna(0).clip(RETURN_CLIP_LOWER, RETURN_CLIP_UPPER)
        df['ret5'] = df['close'].pct_change(RET5_PERIOD).fillna(0).clip(RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER)
        df['ret10'] = df['close'].pct_change(RET10_PERIOD).fillna(0).clip(RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER)
        df['ret15'] = df['close'].pct_change(RET15_PERIOD).fillna(0).clip(RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER)
        df['ret20'] = df['close'].pct_change(RET20_PERIOD).fillna(0).clip(RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER)
        df['ret30'] = df['close'].pct_change(RET30_PERIOD).fillna(0).clip(RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER)
        df['ret60'] = df['close'].pct_change(RET60_PERIOD).fillna(0).clip(RETURN_5D_CLIP_LOWER, RETURN_5D_CLIP_UPPER)

        # Cumulative returns
        df['ret_cum5'] = df['ret'].rolling(5, min_periods=1).sum().fillna(0)
        df['ret_cum10'] = df['ret'].rolling(10, min_periods=1).sum().fillna(0)
        df['ret_cum20'] = df['ret'].rolling(20, min_periods=1).sum().fillna(0)
        df['ret_cum_max'] = df['ret'].rolling(20, min_periods=1).max().fillna(0)

        # ============================================================
        # SECTION 2: MOVING AVERAGE FEATURES (20 features)
        # ============================================================
        df['ma3'] = df['close'].rolling(MA3_PERIOD, min_periods=1).mean()
        df['ma5'] = df['close'].rolling(MA5_PERIOD, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(MA10_PERIOD, min_periods=1).mean()
        df['ma15'] = df['close'].rolling(MA15_PERIOD, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(MA20_PERIOD, min_periods=1).mean()
        df['ma30'] = df['close'].rolling(MA30_PERIOD, min_periods=1).mean()
        df['ma40'] = df['close'].rolling(MA40_PERIOD, min_periods=1).mean()
        df['ma60'] = df['close'].rolling(MA60_PERIOD, min_periods=1).mean()
        df['ma120'] = df['close'].rolling(MA120_PERIOD, min_periods=1).mean()

        # EMA features
        df['ema12'] = FeatureEngineer._calculate_ema(df['close'], EMA_PERIOD)
        df['ema26'] = FeatureEngineer._calculate_ema(df['close'], EMA_LONG_PERIOD)

        # Price relative to MA (trend indicators)
        df['trend'] = (df['close'] / df['ma60'] - 1).fillna(0).clip(TREND_CLIP_LOWER, TREND_CLIP_UPPER)
        df['trend20'] = (df['close'] / df['ma20'] - 1).fillna(0).clip(TREND_CLIP_LOWER, TREND_CLIP_UPPER)
        df['trend120'] = (df['close'] / df['ma120'] - 1).fillna(0).clip(TREND_CLIP_LOWER, TREND_CLIP_UPPER)

        # MA crossover signals
        df['ma_ratio_3_10'] = (df['ma3'] / df['ma10'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['ma_ratio_5_20'] = (df['ma5'] / df['ma20'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['ma_ratio_10_30'] = (df['ma10'] / df['ma30'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['ma_ratio_10_60'] = (df['ma10'] / df['ma60'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['ma_ratio_20_60'] = (df['ma20'] / df['ma60'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['ma_ratio_20_120'] = (df['ma20'] / df['ma120'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['ema_ratio'] = (df['ema12'] / df['ema26'] - 1).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)

        # Distance from MAs
        df['dist_ma5'] = ((df['close'] - df['ma5']) / df['ma5']).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['dist_ma20'] = ((df['close'] - df['ma20']) / df['ma20']).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)
        df['dist_ma60'] = ((df['close'] - df['ma60']) / df['ma60']).fillna(0).clip(MA_RATIO_CLIP_LOWER, MA_RATIO_CLIP_UPPER)

        # ============================================================
        # SECTION 3: MOMENTUM FEATURES (10 features)
        # ============================================================
        df['mom3'] = df['close'].pct_change(MOM3_PERIOD).shift(1).fillna(0)
        df['mom5'] = df['close'].pct_change(MOM5_PERIOD).shift(1).fillna(0)
        df['mom10'] = df['close'].pct_change(MOM10_PERIOD).shift(1).fillna(0)
        df['mom15'] = df['close'].pct_change(MOM15_PERIOD).shift(1).fillna(0)
        df['mom20'] = df['close'].pct_change(MOM20_PERIOD).shift(1).fillna(0)

        # Momentum acceleration (2nd order)
        df['mom_accel'] = (df['mom5'] - df['mom10']).fillna(0)
        df['mom_accel2'] = (df['mom10'] - df['mom20']).fillna(0)

        # ROC (Rate of Change)
        df['roc5'] = ((df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + EPSILON)).fillna(0)
        df['roc10'] = ((df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + EPSILON)).fillna(0)
        df['roc20'] = ((df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + EPSILON)).fillna(0)

        # ============================================================
        # SECTION 4: VOLATILITY FEATURES (12 features)
        # ============================================================
        df['std10'] = df['close'].rolling(STD_SHORT_PERIOD, min_periods=1).std()
        df['std20'] = df['close'].rolling(STD_PERIOD, min_periods=1).std()
        df['std40'] = df['close'].rolling(STD_LONG_PERIOD, min_periods=1).std()

        df['volatility'] = (df['std20'] / df['ma20']).fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)
        df['volatility_short'] = (df['std10'] / df['ma10']).fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)
        df['volatility_long'] = (df['std40'] / df['ma40']).fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)

        # Volatility of volatility
        df['vvol'] = df['volatility'].rolling(10, min_periods=1).std().fillna(0)

        # ATR and ATR ratio
        df['atr'] = FeatureEngineer._calculate_atr(df, ATR_PERIOD)
        df['atr_short'] = FeatureEngineer._calculate_atr(df, ATR_SHORT_PERIOD)
        df['atr_long'] = FeatureEngineer._calculate_atr(df, ATR_LONG_PERIOD)
        df['atr_ratio'] = (df['atr'] / df['close']).fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)
        df['atr_ratio_short'] = (df['atr_short'] / df['close']).fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)
        df['atr_ratio_long'] = (df['atr_long'] / df['close']).fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)

        # Parkinson volatility estimator (if high/low available)
        if has_highlow:
            hl_ratio = np.log(df['high'] / (df['low'] + EPSILON))
            df['parkinson_vol'] = np.sqrt((hl_ratio ** 2).rolling(10, min_periods=1).mean() / (4 * np.log(2)))
            df['parkinson_vol'] = df['parkinson_vol'].fillna(0).clip(VOLATILITY_CLIP_LOWER, VOLATILITY_CLIP_UPPER)

        # ============================================================
        # SECTION 5: VOLUME FEATURES (10 features)
        # ============================================================
        df['volume_ma5'] = df['volume'].rolling(MA5_PERIOD, min_periods=1).mean()
        df['volume_ma10'] = df['volume'].rolling(MA10_PERIOD, min_periods=1).mean()
        df['volume_ma20'] = df['volume'].rolling(MA20_PERIOD, min_periods=1).mean()
        df['volume_ma60'] = df['volume'].rolling(MA60_PERIOD, min_periods=1).mean()

        df['vol_chg'] = (df['volume'] / df['volume_ma20'] - 1).fillna(0).clip(VOL_CHG_CLIP_LOWER, VOL_CHG_CLIP_UPPER)
        df['vol_chg_short'] = (df['volume'] / df['volume_ma5'] - 1).fillna(0).clip(VOL_CHG_CLIP_LOWER, VOL_CHG_CLIP_UPPER)
        df['vol_chg_long'] = (df['volume'] / df['volume_ma60'] - 1).fillna(0).clip(VOL_CHG_CLIP_LOWER, VOL_CHG_CLIP_UPPER)

        # Volume momentum
        df['vol_mom5'] = df['volume'].pct_change(5).fillna(0).clip(VOL_CHG_CLIP_LOWER, VOL_CHG_CLIP_UPPER)
        df['vol_mom10'] = df['volume'].pct_change(10).fillna(0).clip(VOL_CHG_CLIP_LOWER, VOL_CHG_CLIP_UPPER)

        # Volume-price relationships
        df['vol_price_trend'] = df['ret'] * df['vol_chg'].fillna(0)
        df['vol_price_corr'] = (df['ret'].rolling(10, min_periods=1).corr(df['vol_chg'])).fillna(0)

        # Volume volatility
        df['vol_volatility'] = (df['volume'].rolling(20, min_periods=1).std() /
                                (df['volume_ma20'] + EPSILON)).fillna(0)

        # ============================================================
        # SECTION 6: RSI FEATURES (5 features)
        # ============================================================
        df['rsi'] = FeatureEngineer._calculate_rsi(df['close'], RSI_PERIOD)
        df['rsi_short'] = FeatureEngineer._calculate_rsi(df['close'], RSI_SHORT_PERIOD)
        df['rsi_long'] = FeatureEngineer._calculate_rsi(df['close'], RSI_LONG_PERIOD)

        df['rsi_norm'] = ((df['rsi'] - 50) / 50).fillna(0).clip(RSI_CLIP_LOWER, RSI_CLIP_UPPER)
        df['rsi_delta'] = (df['rsi'] - df['rsi_short']).fillna(0).clip(RSI_CLIP_LOWER, RSI_CLIP_UPPER)

        # ============================================================
        # SECTION 7: BOLLINGER BANDS FEATURES (6 features)
        # ============================================================
        bb_upper, bb_middle, bb_lower = FeatureEngineer._calculate_bollinger_bands(df['close'], BOLLINGER_PERIOD)

        df['bb_width'] = ((bb_upper - bb_lower) / (bb_middle + EPSILON)).fillna(0).clip(BOLLINGER_CLIP_LOWER, BOLLINGER_CLIP_UPPER)
        df['bb_position'] = ((df['close'] - bb_lower) / (bb_upper - bb_lower + EPSILON) - 0.5).fillna(0).clip(BOLLINGER_CLIP_LOWER, BOLLINGER_CLIP_UPPER)
        df['bb_upper_dist'] = ((bb_upper - df['close']) / df['close']).fillna(0).clip(BOLLINGER_CLIP_LOWER, BOLLINGER_CLIP_UPPER)
        df['bb_lower_dist'] = ((df['close'] - bb_lower) / df['close']).fillna(0).clip(BOLLINGER_CLIP_LOWER, BOLLINGER_CLIP_UPPER)

        # Bollinger Band squeeze
        df['bb_squeeze'] = (df['bb_width'].rolling(20, min_periods=1).mean() /
                           (df['bb_width'] + EPSILON)).fillna(0)

        # Bollinger Band bandwidth change
        df['bb_width_delta'] = df['bb_width'].diff().fillna(0)

        # ============================================================
        # SECTION 8: MACD FEATURES (5 features)
        # ============================================================
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = FeatureEngineer._calculate_ema(df['macd'], MACD_SIGNAL_PERIOD)
        df['macd_hist'] = df['macd'] - df['macd_signal']

        df['macd_norm'] = (df['macd'] / (df['close'] + EPSILON)).fillna(0).clip(MACD_CLIP_LOWER, MACD_CLIP_UPPER)
        df['macd_hist_norm'] = (df['macd_hist'] / (df['close'] + EPSILON)).fillna(0).clip(MACD_CLIP_LOWER, MACD_CLIP_UPPER)

        # ============================================================
        # SECTION 9: STOCHASTIC FEATURES (4 features)
        # ============================================================
        if has_highlow:
            stoch_k, stoch_d = FeatureEngineer._calculate_stochastic(df, STOCH_K_PERIOD, STOCH_D_PERIOD)
            df['stoch_k'] = ((stoch_k - 50) / 50).fillna(0).clip(STOCH_CLIP_LOWER, STOCH_CLIP_UPPER)
            df['stoch_d'] = ((stoch_d - 50) / 50).fillna(0).clip(STOCH_CLIP_LOWER, STOCH_CLIP_UPPER)
            df['stoch_k_d_diff'] = (df['stoch_k'] - df['stoch_d']).fillna(0)
            df['stoch_k_lag'] = df['stoch_k'].shift(1).fillna(0)

        # ============================================================
        # SECTION 10: HIGH-LOW FEATURES (if available, 8 features)
        # ============================================================
        if has_highlow:
            # Intraday range
            df['highlow_ratio'] = ((df['high'] - df['low']) / df['close']).fillna(0).clip(HIGHLOW_CLIP_LOWER, HIGHLOW_CLIP_UPPER)

            # Close position within day's range
            day_range = (df['high'] - df['low']).replace(0, EPSILON)
            df['close_pos'] = ((df['close'] - df['low']) / day_range - 0.5).fillna(0).clip(HIGHLOW_CLIP_LOWER, HIGHLOW_CLIP_UPPER)

            # Gap from previous close
            prev_close = df['close'].shift(1)
            df['gap'] = ((df['open'] - prev_close) / prev_close if 'open' in df.columns
                        else (df['close'] - prev_close) / prev_close).fillna(0).clip(GAP_CLIP_LOWER, GAP_CLIP_UPPER)

            # High/Low relative to close
            df['high_dist'] = ((df['high'] - df['close']) / df['close']).fillna(0).clip(HIGHLOW_CLIP_LOWER, HIGHLOW_CLIP_UPPER)
            df['low_dist'] = ((df['close'] - df['low']) / df['close']).fillna(0).clip(HIGHLOW_CLIP_LOWER, HIGHLOW_CLIP_UPPER)

            # Williams %R
            df['williams_r'] = (FeatureEngineer._calculate_williams_r(df) / 100).fillna(0).clip(HIGHLOW_CLIP_LOWER, HIGHLOW_CLIP_UPPER)

            # CCI (Commodity Channel Index)
            df['cci'] = np.tanh(FeatureEngineer._calculate_cci(df) / 100).fillna(0)

        else:
            df['gap'] = df['close'].diff().fillna(0).clip(GAP_CLIP_LOWER, GAP_CLIP_UPPER)

        # ============================================================
        # SECTION 11: LAGGED FEATURES (6 features)
        # ============================================================
        df['ret_lag1'] = df['ret'].shift(1).fillna(0)
        df['ret_lag3'] = df['ret'].shift(3).fillna(0)
        df['ret_lag5'] = df['ret'].shift(5).fillna(0)
        df['ret_lag10'] = df['ret'].shift(10).fillna(0)
        df['vol_chg_lag1'] = df['vol_chg'].shift(1).fillna(0)
        df['vol_chg_lag5'] = df['vol_chg'].shift(5).fillna(0)

        # ============================================================
        # SECTION 12: ROLLING STATISTICS (8 features)
        # ============================================================
        # Rolling max/min returns
        df['ret_max_5'] = df['ret'].rolling(5, min_periods=1).max().fillna(0)
        df['ret_min_5'] = df['ret'].rolling(5, min_periods=1).min().fillna(0)
        df['ret_max_10'] = df['ret'].rolling(10, min_periods=1).max().fillna(0)
        df['ret_min_10'] = df['ret'].rolling(10, min_periods=1).min().fillna(0)

        df['ret_range_5'] = (df['ret_max_5'] - df['ret_min_5']).fillna(0)
        df['ret_range_10'] = (df['ret_max_10'] - df['ret_min_10']).fillna(0)

        # Return center (deviation from rolling mean)
        df['ret_center'] = (df['ret'] - df['ret'].rolling(20, min_periods=1).mean()).fillna(0)
        df['ret_center_short'] = (df['ret'] - df['ret'].rolling(5, min_periods=1).mean()).fillna(0)

        # ============================================================
        # SECTION 13: SKEWNESS & KURTOSIS (4 features)
        # ============================================================
        df['ret_skew_10'] = FeatureEngineer._calculate_skewness(df['ret'], 10).fillna(0).clip(SKEW_CLIP_LOWER, SKEW_CLIP_UPPER)
        df['ret_skew_20'] = FeatureEngineer._calculate_skewness(df['ret'], 20).fillna(0).clip(SKEW_CLIP_LOWER, SKEW_CLIP_UPPER)
        df['ret_kurt_10'] = FeatureEngineer._calculate_kurtosis(df['ret'], 10).fillna(0).clip(KURT_CLIP_LOWER, KURT_CLIP_UPPER)
        df['ret_kurt_20'] = FeatureEngineer._calculate_kurtosis(df['ret'], 20).fillna(0).clip(KURT_CLIP_LOWER, KURT_CLIP_UPPER)

        # ============================================================
        # SECTION 14: MONEY FLOW INDEX (2 features)
        # ============================================================
        df['mfi'] = ((FeatureEngineer._calculate_money_flow_index(df) - 50) / 50).fillna(0).clip(RSI_CLIP_LOWER, RSI_CLIP_UPPER)
        df['mfi_delta'] = df['mfi'].diff().fillna(0)

        # ============================================================
        # NORMALIZE ALL FEATURES
        # ============================================================
        features = [
            # Returns (12)
            'ret', 'ret3', 'ret5', 'ret10', 'ret15', 'ret20', 'ret30', 'ret60',
            'ret_cum5', 'ret_cum10', 'ret_cum20', 'ret_cum_max',
            # MA ratios & trends (20)
            'trend', 'trend20', 'trend120',
            'ma_ratio_3_10', 'ma_ratio_5_20', 'ma_ratio_10_30', 'ma_ratio_10_60',
            'ma_ratio_20_60', 'ma_ratio_20_120', 'ema_ratio',
            'dist_ma5', 'dist_ma20', 'dist_ma60',
            # Momentum (10)
            'mom3', 'mom5', 'mom10', 'mom15', 'mom20',
            'mom_accel', 'mom_accel2', 'roc5', 'roc10', 'roc20',
            # Volatility (12)
            'volatility', 'volatility_short', 'volatility_long', 'vvol',
            'atr_ratio', 'atr_ratio_short', 'atr_ratio_long',
            # Volume (10)
            'vol_chg', 'vol_chg_short', 'vol_chg_long',
            'vol_mom5', 'vol_mom10', 'vol_price_trend', 'vol_price_corr', 'vol_volatility',
            # RSI (5)
            'rsi_norm', 'rsi_delta',
            # Bollinger (6)
            'bb_width', 'bb_position', 'bb_upper_dist', 'bb_lower_dist', 'bb_squeeze', 'bb_width_delta',
            # MACD (5)
            'macd_norm', 'macd_hist_norm',
            # Stochastic (4) - conditional
            # High-Low (8) - conditional
            # Gap (1)
            'gap',
            # Lagged (6)
            'ret_lag1', 'ret_lag3', 'ret_lag5', 'ret_lag10',
            'vol_chg_lag1', 'vol_chg_lag5',
            # Rolling stats (8)
            'ret_range_5', 'ret_range_10', 'ret_center', 'ret_center_short',
            # Skew/Kurt (4)
            'ret_skew_10', 'ret_skew_20', 'ret_kurt_10', 'ret_kurt_20',
            # MFI (2)
            'mfi', 'mfi_delta',
        ]

        # Add high-low dependent features
        if has_highlow:
            features.extend([
                'highlow_ratio', 'close_pos', 'high_dist', 'low_dist',
                'williams_r', 'cci', 'parkinson_vol',
                'stoch_k', 'stoch_d', 'stoch_k_d_diff', 'stoch_k_lag',
            ])

        feature_cols = []

        for col in features:
            if col not in df.columns:
                continue
            roll_mean = df[col].rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean().fillna(0)
            roll_std = df[col].rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std().fillna(1)
            df[f'{col}_norm'] = (df[col] - roll_mean) / (roll_std + EPSILON)
            df[f'{col}_norm'] = df[f'{col}_norm'].clip(FEATURE_CLIP_LOWER, FEATURE_CLIP_UPPER)
            feature_cols.append(f'{col}_norm')

        df = df.ffill().fillna(0)

        logger.info(f"创建了 {len(feature_cols)} 个特征")
        return df, feature_cols
