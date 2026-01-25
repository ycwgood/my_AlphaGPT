"""Factor operators definition"""

import numpy as np
from typing import List, Tuple, Callable, Dict


class FactorOperators:
    """Define operators for factor generation"""

    # 一元算子注册表
    _UNARY_OPERATORS: Dict[str, Callable] = {
        "NEG": lambda x: -x,
        "ABS": lambda x: np.abs(x),
        "DELTA": lambda x: np.concatenate([np.zeros(1, dtype=x.dtype), x[1:] - x[:-1]]),
        "SIGN": lambda x: np.sign(x),
        "LOG": lambda x: np.log(np.abs(x) + 1e-6),
        "SQRT": lambda x: np.sqrt(np.abs(x)),
    }

    # 二元算子注册表
    _BINARY_OPERATORS: Dict[str, Callable] = {
        "ADD": lambda x, y: x + y,
        "SUB": lambda x, y: x - y,
        "MUL": lambda x, y: x * y,
        "DIV": lambda x, y: x / (np.abs(y) + 1e-6),
        "MAX": lambda x, y: np.maximum(x, y),
        "MIN": lambda x, y: np.minimum(x, y),
    }

    @classmethod
    def get_unary_operators(cls) -> List[str]:
        """获取所有一元算子名称"""
        return list(cls._UNARY_OPERATORS.keys())

    @classmethod
    def get_binary_operators(cls) -> List[str]:
        """获取所有二元算子名称"""
        return list(cls._BINARY_OPERATORS.keys())

    @classmethod
    def apply_unary(cls, op_name: str, operand: np.ndarray) -> np.ndarray:
        """
        应用一元算子

        Args:
            op_name: 算子名称
            operand: 操作数数组

        Returns:
            计算结果数组

        Raises:
            ValueError: 未知算子
        """
        if op_name not in cls._UNARY_OPERATORS:
            raise ValueError(f"Unknown unary operator: {op_name}")
        return cls._UNARY_OPERATORS[op_name](operand)

    @classmethod
    def apply_binary(cls, op_name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        应用二元算子

        Args:
            op_name: 算子名称
            x: 第一个操作数
            y: 第二个操作数

        Returns:
            计算结果数组

        Raises:
            ValueError: 未知算子
        """
        if op_name not in cls._BINARY_OPERATORS:
            raise ValueError(f"Unknown binary operator: {op_name}")
        return cls._BINARY_OPERATORS[op_name](x, y)

    @classmethod
    def is_unary(cls, op_name: str) -> bool:
        """判断是否为一元算子"""
        return op_name in cls._UNARY_OPERATORS

    @classmethod
    def is_binary(cls, op_name: str) -> bool:
        """判断是否为二元算子"""
        return op_name in cls._BINARY_OPERATORS

    @staticmethod
    def get_operators() -> List[Tuple[str, Callable, int]]:
        """
        Get list of available operators (兼容旧接口)

        Returns:
            List of tuples (operator_name, operator_function, arity)
        """
        ops = []
        for name, func in FactorOperators._BINARY_OPERATORS.items():
            ops.append((name, func, 2))
        for name, func in FactorOperators._UNARY_OPERATORS.items():
            ops.append((name, func, 1))
        return ops

    @staticmethod
    def get_operator_descriptions() -> List[str]:
        """Get human-readable operator descriptions"""
        descriptions = []

        # 二元算子描述
        binary_desc = {
            "ADD": "加法 A + B",
            "SUB": "减法 A - B",
            "MUL": "乘法 A * B",
            "DIV": "除法 A / B（自动防除零）",
            "MAX": "取最大值 max(A, B)",
            "MIN": "取最小值 min(A, B)",
        }
        for name in FactorOperators._BINARY_OPERATORS:
            desc = binary_desc.get(name, f"{name}(A, B)")
            descriptions.append(f"{name}: 二元算子，{desc}")

        # 一元算子描述
        unary_desc = {
            "NEG": "取负 -A",
            "ABS": "绝对值 |A|",
            "DELTA": "一阶差分 A[t] - A[t-1]",
            "SIGN": "符号函数 sign(A)",
            "LOG": "对数 log(|A|)",
            "SQRT": "平方根 sqrt(|A|)",
        }
        for name in FactorOperators._UNARY_OPERATORS:
            desc = unary_desc.get(name, f"{name}(A)")
            descriptions.append(f"{name}: 一元算子，{desc}")

        return descriptions
