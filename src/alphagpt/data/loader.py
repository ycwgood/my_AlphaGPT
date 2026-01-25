"""Data loading module - Tushare version"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from alphagpt.utils.exceptions import DataLoadError

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare price data using Tushare"""

    @staticmethod
    def get_price_data(
        ts_code: str,
        start: str,
        end: str,
        tushare_token: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get price data from Tushare or generate mock data

        Args:
            ts_code: 股票代码，格式如 000001.SZ
            start: 开始日期 (YYYY-MM-DD 或 YYYYMMDD)
            end: 结束日期 (YYYY-MM-DD 或 YYYYMMDD)
            tushare_token: Tushare API Token

        Returns:
            DataFrame with price data or None if failed
        """
        if tushare_token:
            return DataLoader._get_tushare_data(ts_code, start, end, tushare_token)
        else:
            logger.warning("Tushare Token 未配置，使用模拟数据")
            return DataLoader._generate_mock_data(start, end)

    @staticmethod
    def _get_tushare_data(
        ts_code: str,
        start: str,
        end: str,
        token: str
    ) -> Optional[pd.DataFrame]:
        """
        使用 Tushare 获取历史数据

        Args:
            ts_code: 股票代码，如 000001.SZ
            start: 开始日期
            end: 结束日期
            token: Tushare API Token

        Returns:
            DataFrame with price data
        """
        try:
            import tushare as ts

            logger.info(f"使用 Tushare 获取 {ts_code} 从 {start} 到 {end} 的数据...")

            # 设置 token
            ts.set_token(token)
            pro = ts.pro_api()

            # 转换日期格式为 YYYYMMDD
            start_date = start.replace('-', '')
            end_date = end.replace('-', '')

            # 获取日线数据
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                logger.warning("=" * 50)
                logger.warning(f"⚠️  警告: 未获取到 {ts_code} 的真实数据")
                logger.warning("⚠️  将使用模拟数据进行演示，结果仅供参考！")
                logger.warning("=" * 50)
                return DataLoader._generate_mock_data(start, end)

            # 按日期排序（Tushare 返回的数据是倒序的）
            df = df.sort_values('trade_date').reset_index(drop=True)

            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date

            # 重命名列以匹配原有格式
            df = df.rename(columns={'vol': 'volume'})

            # 选择需要的列
            result = df[['trade_date', 'open', 'close', 'high', 'low', 'volume']].copy()

            logger.info(f"成功加载 {ts_code} 数据，共 {len(result)} 行")
            return result

        except ImportError as e:
            logger.error("Tushare 未安装，请运行: pip install tushare")
            raise DataLoadError("Tushare未安装，请运行: pip install tushare") from e
        except Exception as e:
            logger.error(f"Tushare 获取数据失败: {type(e).__name__}: {str(e)}")
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
        ts_codes: List[str],
        start: str,
        end: str,
        tushare_token: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个股票的价格数据

        Args:
            ts_codes: 股票代码列表
            start: 开始日期
            end: 结束日期
            tushare_token: Tushare API Token

        Returns:
            字典，键为股票代码，值为对应的DataFrame
        """
        result = {}
        for ts_code in ts_codes:
            logger.info(f"正在加载 {ts_code} 的数据...")
            try:
                df = DataLoader.get_price_data(ts_code, start, end, tushare_token)
                if df is not None and not df.empty:
                    result[ts_code] = df
                    logger.info(f"成功加载 {ts_code}，共 {len(df)} 行")
                else:
                    logger.warning(f"跳过 {ts_code}，数据为空")
            except Exception as e:
                logger.error(f"加载 {ts_code} 失败: {e}")
                continue

        if not result:
            raise DataLoadError("所有股票数据加载失败")

        logger.info(f"批量加载完成，成功加载 {len(result)}/{len(ts_codes)} 个股票")
        return result
