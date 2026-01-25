"""Factor calculator module"""

import logging
import numpy as np
import numexpr as ne
from typing import Dict, Optional
import re
from alphagpt.utils.validators import validate_array_lengths
from alphagpt.utils.constants import (
    MIN_SAMPLE_SIZE, EPSILON, POSITION_SCALE, TRADING_DAYS_PER_YEAR,
    FACTOR_CLIP_MIN, FACTOR_CLIP_MAX
)
from .operators import FactorOperators

logger = logging.getLogger(__name__)


class FactorCalculator:
    """Calculate factor values from expressions"""

    @staticmethod
    def _process_unary_operators(
        expr: str,
        local_vars: Dict[str, np.ndarray]
    ) -> str:
        """
        处理表达式中的所有一元算子（支持嵌套和复杂操作数）

        Args:
            expr: 因子表达式
            local_vars: 变量字典（会被修改）

        Returns:
            处理后的表达式
        """
        unary_ops = FactorOperators.get_unary_operators()

        # 循环处理直到没有更多算子可处理
        max_iterations = 100  # 防止无限循环
        for _ in range(max_iterations):
            found = False
            for op_name in unary_ops:
                # 匹配不包含括号的最内层操作数
                pattern = rf'{op_name}\(([^()]+)\)'
                match = re.search(pattern, expr)
                if match:
                    operand_expr = match.group(1).strip()

                    # 尝试计算操作数的值
                    try:
                        if operand_expr in local_vars:
                            # 简单变量名
                            operand_value = local_vars[operand_expr]
                        else:
                            # 复杂表达式，用 numexpr 计算
                            operand_value = ne.evaluate(operand_expr, local_dict=local_vars)

                        # 应用算子
                        temp_var = f'_temp_{len(local_vars)}'
                        local_vars[temp_var] = FactorOperators.apply_unary(op_name, operand_value)
                        expr = expr.replace(match.group(0), temp_var, 1)
                        found = True
                        break
                    except Exception:
                        # 无法计算，跳过这个匹配
                        continue

            if not found:
                break

        return expr

    @staticmethod
    def _convert_prefix_notation(expr: str) -> str:
        """
        将前缀表示法转换为中缀表示法
        例如: +(a, b) -> (a + b)

        Args:
            expr: 可能包含前缀表示法的表达式

        Returns:
            转换后的中缀表示法表达式
        """
        prefix_ops = {'+': '+', '-': '-', '*': '*', '/': '/'}

        max_iterations = 100
        for _ in range(max_iterations):
            found = False
            for op_char, op_symbol in prefix_ops.items():
                escaped_op = re.escape(op_char)
                pattern = rf'{escaped_op}\(([^(),]+),\s*([^()]+)\)'
                match = re.search(pattern, expr)
                if match:
                    arg1 = match.group(1).strip()
                    arg2 = match.group(2).strip()
                    replacement = f'({arg1} {op_symbol} {arg2})'
                    expr = expr[:match.start()] + replacement + expr[match.end():]
                    found = True
                    break
            if not found:
                break

        return expr

    @staticmethod
    def _process_binary_operators(
        expr: str,
        local_vars: Dict[str, np.ndarray]
    ) -> str:
        """
        处理表达式中的二元算子（如 MAX, MIN）

        Args:
            expr: 因子表达式
            local_vars: 变量字典（会被修改）

        Returns:
            处理后的表达式
        """
        binary_ops = FactorOperators.get_binary_operators()
        # 只处理函数形式的二元算子（MAX, MIN）
        func_binary_ops = [op for op in binary_ops if op in ('MAX', 'MIN')]

        max_iterations = 100
        for _ in range(max_iterations):
            found = False
            for op_name in func_binary_ops:
                # 匹配 OP(arg1, arg2) 形式，arg1和arg2不包含括号
                pattern = rf'{op_name}\(([^(),]+),\s*([^()]+)\)'
                match = re.search(pattern, expr)
                if match:
                    arg1_expr = match.group(1).strip()
                    arg2_expr = match.group(2).strip()

                    try:
                        # 计算两个操作数
                        if arg1_expr in local_vars:
                            arg1_value = local_vars[arg1_expr]
                        else:
                            arg1_value = ne.evaluate(arg1_expr, local_dict=local_vars)

                        if arg2_expr in local_vars:
                            arg2_value = local_vars[arg2_expr]
                        else:
                            arg2_value = ne.evaluate(arg2_expr, local_dict=local_vars)

                        # 应用算子
                        temp_var = f'_temp_{len(local_vars)}'
                        local_vars[temp_var] = FactorOperators.apply_binary(op_name, arg1_value, arg2_value)
                        expr = expr.replace(match.group(0), temp_var, 1)
                        found = True
                        break
                    except Exception:
                        continue

            if not found:
                break

        return expr

    @staticmethod
    def calculate_factor_value(
        expr: str,
        feature_data: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Calculate factor value from expression using safe evaluation

        Args:
            expr: Factor expression string
            feature_data: Dictionary of feature name to array

        Returns:
            Factor value array or None if calculation failed
        """
        try:
            # Input validation
            if not expr or not isinstance(expr, str):
                logger.error("表达式必须是非空字符串")
                return None

            if not feature_data:
                logger.error("特征数据不能为空")
                return None

            # Validate array lengths are consistent
            validate_array_lengths(feature_data)

            # Process operators in expression
            local_vars = feature_data.copy()
            # 先转换前缀表示法为中缀表示法
            processed_expr = FactorCalculator._convert_prefix_notation(expr)
            # 交替处理一元和二元算子，直到没有变化
            max_iterations = 50
            for _ in range(max_iterations):
                prev_expr = processed_expr
                # 先处理一元算子
                processed_expr = FactorCalculator._process_unary_operators(processed_expr, local_vars)
                # 再处理二元算子
                processed_expr = FactorCalculator._process_binary_operators(processed_expr, local_vars)
                if processed_expr == prev_expr:
                    break

            # Use numexpr to safely evaluate the final expression
            factor_value = ne.evaluate(processed_expr, local_dict=local_vars)

            # Clean outliers
            factor_value = np.nan_to_num(factor_value)
            factor_value = np.clip(factor_value, FACTOR_CLIP_MIN, FACTOR_CLIP_MAX)

            return factor_value

        except Exception as e:
            logger.error(f"计算因子失败: {expr}, 错误: {type(e).__name__}: {str(e)}")
            return None

    @staticmethod
    def calculate_sharpe(
        factor_values: np.ndarray,
        target_returns: np.ndarray,
        position_scale: float = 0.3
    ) -> float:
        """
        Calculate Sharpe ratio for factor

        Args:
            factor_values: Factor value array
            target_returns: Target return array
            position_scale: Position scaling factor

        Returns:
            Sharpe ratio
        """
        if len(factor_values) < MIN_SAMPLE_SIZE or len(target_returns) < MIN_SAMPLE_SIZE:
            return -np.inf

        # Ensure same length
        min_len = min(len(factor_values), len(target_returns))
        factor_values = factor_values[:min_len]
        target_returns = target_returns[:min_len]

        # Normalize factor values
        factor_norm = (factor_values - np.mean(factor_values)) / (np.std(factor_values) + EPSILON)

        # Calculate positions
        position = np.tanh(factor_norm * position_scale)

        # Calculate strategy returns (t时刻仓位 × t+1时刻收益)
        strategy_returns = position[:-1] * target_returns[1:]

        if len(strategy_returns) < MIN_SAMPLE_SIZE or np.std(strategy_returns) < EPSILON:
            return -np.inf

        # Calculate Sharpe ratio
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return sharpe

    @staticmethod
    def evaluate_factor(
        factor_values: np.ndarray,
        target_returns: np.ndarray,
        position_scale: float = 0.3,
        lookback: int = 60
    ) -> Dict:
        """
        Unified factor evaluation method using rolling window normalization
        to avoid look-ahead bias (consistent with BacktestEngine)

        Args:
            factor_values: Factor value array
            target_returns: Target return array
            position_scale: Position scaling factor
            lookback: Rolling window size for normalization

        Returns:
            Dictionary with evaluation metrics (sharpe, ic, returns, positions)
        """
        import pandas as pd
        from alphagpt.backtest.engine import BacktestEngine

        if len(factor_values) < lookback or len(target_returns) < lookback:
            return {
                'sharpe': -np.inf,
                'ic': np.nan,
                'returns': np.array([]),
                'positions': np.array([])
            }

        # Use rolling window normalization with shift to avoid look-ahead bias
        factor_series = pd.Series(factor_values)
        rolling_mean = factor_series.rolling(
            window=lookback,
            min_periods=lookback // 2
        ).mean().shift(1)
        rolling_std = factor_series.rolling(
            window=lookback,
            min_periods=lookback // 2
        ).std().shift(1)

        # Normalize using shifted rolling stats
        norm = (factor_values - rolling_mean.values) / (rolling_std.values + EPSILON)
        position = np.tanh(norm * position_scale)
        position[:lookback] = 0  # warmup period
        position = np.nan_to_num(position, nan=0.0)

        # Calculate strategy returns
        min_len = min(len(position) - 1, len(target_returns))
        position = position[:min_len + 1]
        target_returns_adj = target_returns[:min_len]
        strategy_returns = position[:-1] * target_returns_adj

        if len(strategy_returns) < MIN_SAMPLE_SIZE:
            return {
                'sharpe': -np.inf,
                'ic': np.nan,
                'returns': strategy_returns,
                'positions': position
            }

        # Calculate Sharpe ratio
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Calculate IC
        ic = BacktestEngine.calculate_ic(
            factor_values[:len(strategy_returns)],
            target_returns_adj[:len(strategy_returns)]
        )

        return {
            'sharpe': sharpe,
            'ic': ic,
            'returns': strategy_returns,
            'positions': position
        }
