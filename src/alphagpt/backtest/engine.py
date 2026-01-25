"""Backtest engine module"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.stats import spearmanr
from alphagpt.utils.constants import (
    WARMUP_PERIOD, POSITION_SCALE, EPSILON,
    MIN_SAMPLE_SIZE, TRADING_DAYS_PER_YEAR
)

logger = logging.getLogger(__name__)

# Default rolling window size for normalization (trading days)
DEFAULT_ROLLING_WINDOW = 60


class BacktestEngine:
    """Backtest trading strategies based on factors"""

    @staticmethod
    def backtest_strategy(
        factor_values: np.ndarray,
        test_df: pd.DataFrame,
        test_returns: np.ndarray,
        factor_name: str = "因子"
    ) -> Optional[Dict]:
        """
        Backtest a factor-based strategy

        Args:
            factor_values: Factor value array
            test_df: Test DataFrame
            test_returns: Test return array
            factor_name: Name of the factor

        Returns:
            Dictionary with backtest results or None if failed
        """
        if factor_values is None or len(factor_values) != len(test_df):
            logger.error(f"{factor_name}: 因子值与测试集长度不匹配")
            return None

        # Rolling window normalization (avoids look-ahead bias)
        # Use shift(1) to ensure we only use historical data for normalization
        factor_series = pd.Series(factor_values)
        rolling_mean = factor_series.rolling(
            window=DEFAULT_ROLLING_WINDOW,
            min_periods=WARMUP_PERIOD
        ).mean().shift(1)
        rolling_std = factor_series.rolling(
            window=DEFAULT_ROLLING_WINDOW,
            min_periods=WARMUP_PERIOD
        ).std().shift(1)

        # Vectorized normalization using shifted rolling stats
        norm = (factor_values - rolling_mean.values) / (rolling_std.values + EPSILON)
        positions = np.tanh(norm * POSITION_SCALE)

        # Set warmup period to 0
        positions[:DEFAULT_ROLLING_WINDOW] = 0
        positions = np.nan_to_num(positions, nan=0.0)

        # Ensure same length
        min_len = min(len(positions) - 1, len(test_returns))
        positions = positions[:min_len + 1]
        test_returns_adj = test_returns[:min_len]

        # Strategy returns
        strategy_returns = positions[:-1] * test_returns_adj

        if len(strategy_returns) < MIN_SAMPLE_SIZE:
            logger.warning(f"{factor_name}: 策略收益数据不足 ({len(strategy_returns)} 个样本)")
            return None

        # Calculate IC (Information Coefficient)
        ic = BacktestEngine.calculate_ic(
            factor_values[:len(strategy_returns)],
            test_returns_adj[:len(strategy_returns)]
        )

        # Calculate metrics
        cumulative = (1 + strategy_returns).cumprod()
        total_return = cumulative[-1] - 1

        days = len(strategy_returns)
        annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1
        annual_vol = np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = annual_return / (annual_vol + EPSILON)

        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)

        # Benchmark
        benchmark_returns = test_returns_adj
        benchmark_cumulative = (1 + benchmark_returns).cumprod()

        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        info_ratio = np.mean(excess_returns) / (np.std(excess_returns) + EPSILON) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Win rate
        win_rate = np.mean(strategy_returns > 0) * 100

        # Calmar ratio
        if max_dd < 0:
            calmar = annual_return / abs(max_dd)
        else:
            calmar = 0

        result = {
            'factor_name': factor_name,
            'sharpe': sharpe,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'info_ratio': info_ratio,
            'win_rate': win_rate,
            'calmar': calmar,
            'ic': ic,
            'positions': positions,
            'strategy_returns': strategy_returns,
            'cumulative': cumulative,
            'benchmark_cumulative': benchmark_cumulative,
            'drawdown': drawdown
        }

        return result

    @staticmethod
    def calculate_ic(
        factor_values: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Calculate Information Coefficient (IC) - Spearman correlation
        between factor values and subsequent returns

        Args:
            factor_values: Factor value array at time t
            returns: Return array at time t+1

        Returns:
            IC value (Spearman correlation coefficient)
        """
        valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
        valid_count = np.sum(valid_mask)

        if valid_count < MIN_SAMPLE_SIZE:
            logger.warning(f"IC计算: 有效样本不足 ({valid_count} < {MIN_SAMPLE_SIZE})")
            return np.nan

        try:
            ic, _ = spearmanr(
                factor_values[valid_mask],
                returns[valid_mask],
                nan_policy='omit'
            )
            return ic if not np.isnan(ic) else 0.0
        except Exception as e:
            logger.error(f"IC计算失败: {e}")
            return np.nan

    @staticmethod
    def calculate_rolling_ic(
        factor_values: np.ndarray,
        returns: np.ndarray,
        window: int = 60
    ) -> np.ndarray:
        """
        Calculate rolling IC to assess factor stability over time

        Args:
            factor_values: Factor value array
            returns: Return array
            window: Rolling window size

        Returns:
            Array of rolling IC values
        """
        rolling_ic = []
        for i in range(window, len(factor_values)):
            ic = BacktestEngine.calculate_ic(
                factor_values[i-window:i],
                returns[i-window:i]
            )
            rolling_ic.append(ic if not np.isnan(ic) else 0.0)
        return np.array(rolling_ic)
