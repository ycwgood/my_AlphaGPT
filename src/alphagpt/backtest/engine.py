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

        # 1. 确保数组长度一致
        min_len = min(len(factor_values), len(test_returns), len(test_df))
        factor_values = factor_values[:min_len]
        test_returns = test_returns[:min_len]
        test_df = test_df.iloc[:min_len].reset_index(drop=True)

        # 2. 滚动归一化（使用移位避免未来函数偏差）
        factor_series = pd.Series(factor_values)
        rolling_mean = factor_series.rolling(
            window=DEFAULT_ROLLING_WINDOW,
            min_periods=WARMUP_PERIOD
        ).mean().shift(1)
        rolling_std = factor_series.rolling(
            window=DEFAULT_ROLLING_WINDOW,
            min_periods=WARMUP_PERIOD
        ).std().shift(1)

        # 处理 NaN 值
        rolling_mean = rolling_mean.fillna(0)
        rolling_std = rolling_std.fillna(1)

        # 3. 标准化因子值
        norm = (factor_values - rolling_mean.values) / (rolling_std.values + EPSILON)
        norm = np.nan_to_num(norm, nan=0.0)
        
        # 4. 生成头寸（值域 [-1, 1]）
        positions = np.tanh(norm * POSITION_SCALE)
        
        # 5. 预热期置零（前60天不交易）
        positions[:DEFAULT_ROLLING_WINDOW] = 0
        
        # 关键修复：确保只在有效期计算收益
        valid_idx = np.arange(DEFAULT_ROLLING_WINDOW, len(positions) - 1)
        
        if len(valid_idx) < MIN_SAMPLE_SIZE:
            logger.warning(f"{factor_name}: 有效交易期不足")
            return None
        
        # 6. 计算策略收益（只用有效期的头寸和收益）
        strategy_positions = positions[valid_idx]
        strategy_returns_adj = test_returns[valid_idx]
        
        # ✅ 关键：逐日收益 = 头寸 * 单日收益率（不是累乘！）
        strategy_returns = strategy_positions * strategy_returns_adj
        
        # 7. 计算IC（使用对应的因子值和收益）
        ic = BacktestEngine.calculate_ic(
            factor_values[valid_idx],
            strategy_returns_adj
        )

        # 8. 计算绩效指标
        # 清理 NaN 和 inf 值
        strategy_returns = np.nan_to_num(strategy_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        cumulative = np.cumprod(1 + strategy_returns)  # 逐日复利
        total_return = cumulative[-1] - 1

        days = len(strategy_returns)
        annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1
        annual_vol = np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = annual_return / (annual_vol + EPSILON)

        # 最大回撤
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + EPSILON)
        max_dd = np.min(drawdown)

        # 基准收益（持有策略）
        benchmark_cumulative = np.cumprod(1 + strategy_returns_adj)
        
        # 信息比率
        excess_returns = strategy_returns - strategy_returns_adj
        info_ratio = np.mean(excess_returns) / (np.std(excess_returns) + EPSILON) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 胜率
        win_rate = np.mean(strategy_returns > 0) * 100

        # 卡玛比率
        calmar = annual_return / abs(max_dd) if max_dd < 0 else 0

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
            'positions': strategy_positions,
            'strategy_returns': strategy_returns,
            'cumulative': cumulative,
            'benchmark_cumulative': benchmark_cumulative,
            'drawdown': drawdown,
            'num_trades': days
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
