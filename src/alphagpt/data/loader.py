"""Data loading module - qlib version"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from alphagpt.utils.exceptions import DataLoadError
import qlib
from qlib.data import D

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare price data using qlib"""

    @staticmethod
    def get_price_data(
        code: str,
        start: str,
        end: str
    ) -> Optional[pd.DataFrame]:
        """
        Get price data from qlib

        Args:
            code: 股票代码，格式如 SZ000001
            start: 开始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with price data or None if failed
        """
        try:
            logger.info(f"使用 qlib 获取 {code} 从 {start} 到 {end} 的数据...")

            # qlib初始化，请根据实际数据路径修改
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

            # 获取日线数据
            df = D.features(
                instruments=['SH600054'], 
                fields=['$open', '$close', '$high', '$low', '$volume'], 
                start_time=start, 
                end_time=end
            )

            if df is None or df.empty:
                logger.warning("=" * 50)
                logger.warning(f"⚠️  警告: 未获取到 {code} 的真实数据")
                logger.warning("⚠️  将使用模拟数据进行演示，结果仅供参考！")
                logger.warning("=" * 50)
                return DataLoader._generate_mock_data(start, end)
            
            df = df.reset_index()

            # 重命名列以匹配原有格式
            df = df.rename(columns={
                'datetime': 'trade_date',
                '$open': 'open',
                '$close': 'close',
                '$high': 'high',
                '$low': 'low',
                '$volume': 'volume'
            })

            # 日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date

            # 选择需要的列
            result = df[['trade_date', 'open', 'close', 'high', 'low', 'volume']].copy()

            logger.info(f"成功加载 {code} 数据，共 {len(result)} 行")
            return result

        except ImportError as e:
            logger.error("qlib 未安装，请运行: pip install pyqlib")
            raise DataLoadError("qlib 未安装，请运行: pip install pyqlib") from e
        except Exception as e:
            logger.error(f"qlib 获取数据失败: {type(e).__name__}: {str(e)}")
            raise DataLoadError(f"数据获取失败: {str(e)}") from e

    @staticmethod
    def _generate_mock_data(start: str, end: str) -> pd.DataFrame:
        """
        生成模拟数据用于测试

        Args:
            start: 开始日期 (YYYY-MM-DD 或 YYYYMMDD)
            end: 结束日期 (YYYY-MM-DD 或 YYYYMMDD)

        Returns:
            DataFrame with mock price data
        """
        logger.info(f"生成模拟数据从 {start} 到 {end}")

        # 处理日期格式
        if '-' not in start:
            start = f"{start[:4]}-{start[4:6]}-{start[6:]}"
        if '-' not in end:
            end = f"{end[:4]}-{end[4:6]}-{end[6:]}"

        dates = pd.date_range(start=start, end=end, freq='D')

        # 生成随机价格数据（模拟股票价格）
        base_price = 10.0  # 股票基准价格
        data = {
            'trade_date': dates.date,
            'open': base_price + np.random.randn(len(dates)).cumsum() * 0.1,
            'close': base_price + np.random.randn(len(dates)).cumsum() * 0.1 + 0.05,
            'high': base_price + np.random.randn(len(dates)).cumsum() * 0.1 + 0.2,
            'low': base_price + np.random.randn(len(dates)).cumsum() * 0.1 - 0.2,
            'volume': np.random.lognormal(15, 1, len(dates))
        }

        df = pd.DataFrame(data)
        logger.info(f"生成模拟数据完成，共 {len(df)} 行")
        return df

    @staticmethod
    def get_multiple_price_data(
        codes: List[str],
        start: str,
        end: str
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个股票的价格数据

        Args:
            codes
            start: 开始日期
            end: 结束日期

        Returns:
            字典，键为股票代码，值为对应的DataFrame
        """
        result = {}
        for code in codes:
            logger.info(f"正在加载 {code} 的数据...")
            try:
                df = DataLoader.get_price_data(code, start, end)
                if df is not None and not df.empty:
                    result[code] = df
                    logger.info(f"成功加载 {code}，共 {len(df)} 行")
                else:
                    logger.warning(f"跳过 {code}，数据为空")
            except Exception as e:
                logger.error(f"加载 {code} 失败: {e}")
                continue

        if not result:
            raise DataLoadError("所有股票数据加载失败")

        logger.info(f"批量加载完成，成功加载 {len(result)}/{len(codes)} 个股票")
        return result
