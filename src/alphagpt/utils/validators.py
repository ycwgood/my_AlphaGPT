"""Data validation utilities for AlphaGPT"""

import pandas as pd
import numpy as np
from typing import List, Dict
from .exceptions import ValidationError


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate DataFrame contains required columns

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValidationError: If validation fails
    """
    if df is None or df.empty:
        raise ValidationError("DataFrame不能为空")

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"缺少必需的列: {missing_cols}")


def validate_price_data(df: pd.DataFrame) -> None:
    """
    Validate price data validity

    Args:
        df: DataFrame with price data

    Raises:
        ValidationError: If validation fails
    """
    if 'close' not in df.columns:
        raise ValidationError("DataFrame必须包含'close'列")

    if df['close'].isna().all():
        raise ValidationError("close列全为NaN")

    if (df['close'] <= 0).any():
        invalid_count = (df['close'] <= 0).sum()
        raise ValidationError(f"检测到{invalid_count}个非正的close价格")


def validate_array_lengths(arrays: Dict[str, np.ndarray]) -> None:
    """
    Validate all arrays have consistent lengths

    Args:
        arrays: Dictionary of array name to array

    Raises:
        ValidationError: If arrays have different lengths
    """
    if not arrays:
        raise ValidationError("数组字典不能为空")

    lengths = {name: len(arr) for name, arr in arrays.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) > 1:
        raise ValidationError(f"数组长度不一致: {lengths}")


def validate_sample_size(data_length: int, min_samples: int, data_name: str = "数据") -> None:
    """
    Validate minimum sample size requirement

    Args:
        data_length: Actual data length
        min_samples: Minimum required samples
        data_name: Name of the data for error message

    Raises:
        ValidationError: If sample size is insufficient
    """
    if data_length < min_samples:
        raise ValidationError(f"{data_name}样本不足: {data_length} < {min_samples}")

