"""Output directory management utilities"""

import os
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


def ensure_output_dir(output_dir: str = "results") -> str:
    """
    确保输出目录存在，如果不存在则创建

    Args:
        output_dir: 输出目录路径

    Returns:
        输出目录的绝对路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")

    return os.path.abspath(output_dir)


def create_run_output_dir(
    base_dir: str = "results",
    ts_codes: Optional[List[str]] = None,
    date_str: Optional[str] = None
) -> str:
    """
    创建运行输出目录，按日期和股票代码组织

    目录结构: results/YYYY-MM-DD/stock_code1_stock_code2/

    Args:
        base_dir: 基础输出目录
        ts_codes: 股票代码列表
        date_str: 日期字符串 (YYYY-MM-DD)，默认为当天

    Returns:
        创建的输出目录绝对路径
    """
    # 使用当前日期或指定日期
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # 创建股票代码标签
    if ts_codes:
        if len(ts_codes) <= 3:
            stock_label = "_".join(ts_codes)
        else:
            stock_label = f"{len(ts_codes)}stocks"
    else:
        stock_label = "unknown"

    # 构建完整路径
    run_dir = os.path.join(base_dir, date_str, stock_label)

    # 创建目录
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"创建运行输出目录: {run_dir}")

    return os.path.abspath(run_dir)


def get_timestamped_filename(base_name: str, extension: str = "csv") -> str:
    """
    生成带时间戳的文件名

    Args:
        base_name: 基础文件名
        extension: 文件扩展名

    Returns:
        带时间戳的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def get_unique_filename(
    base_dir: str,
    base_name: str,
    extension: str = "csv",
    max_attempts: int = 1000
) -> str:
    """
    生成唯一的文件名，如果文件存在则添加数字后缀

    Args:
        base_dir: 基础目录
        base_name: 基础文件名
        extension: 文件扩展名
        max_attempts: 最大尝试次数

    Returns:
        唯一的文件路径
    """
    base_path = os.path.join(base_dir, f"{base_name}.{extension}")

    if not os.path.exists(base_path):
        return base_path

    for i in range(1, max_attempts + 1):
        new_path = os.path.join(base_dir, f"{base_name}_{i}.{extension}")
        if not os.path.exists(new_path):
            return new_path

    # 如果所有尝试都失败，返回带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{base_name}_{timestamp}.{extension}")
