"""
AlphaGPT Tushare 使用示例
演示如何使用 Tushare 获取股票数据并进行因子分析
"""

from alphagpt.config import Config
from alphagpt.utils import setup_logger, set_seed
from alphagpt.data import DataLoader, FeatureEngineer
from alphagpt.factor import FactorGenerator, FactorCalculator
import numpy as np


def tushare_example():
    """Tushare 数据获取示例"""

    print("=" * 60)
    print("AlphaGPT Tushare 使用示例")
    print("=" * 60)

    # 从 YAML 配置文件加载配置
    config = Config.from_yaml('config.yaml')

    # 检查 Tushare Token 是否配置
    if not config.tushare_token or config.tushare_token == "your_tushare_token_here":
        print("\n⚠️  未配置 Tushare Token，将使用模拟数据")
        print("请在 config.yaml 中配置:")
        print("  tushare:")
        print("    token: '你的Tushare Token'")
        print("\n获取 Token: https://tushare.pro/register")
        config.tushare_token = None

    # 设置日志和随机种子
    logger = setup_logger(__name__)
    set_seed(42)

    # 加载数据
    print(f"\n正在加载数据: {config.ts_codes[0]}")
    print(f"日期范围: {config.start_date} 到 {config.end_date}")

    df = DataLoader.get_price_data(
        ts_code=config.ts_codes[0],
        start=config.start_date,
        end=config.end_date,
        tushare_token=config.tushare_token
    )

    if df is None or len(df) == 0:
        print("❌ 数据加载失败")
        return

    print(f"✅ 成功加载 {len(df)} 条数据")
    print(f"\n数据预览:")
    print(df.head())

    # 特征工程
    print("\n正在创建特征...")
    df, features = FeatureEngineer.create_features(df)
    print(f"✅ 创建了 {len(features)} 个特征: {', '.join(features)}")

    # 生成因子（使用本地生成器）
    print("\n正在生成因子...")
    generator = FactorGenerator(use_api=False)
    factor_exprs = generator.generate_factors_local(features, num_factors=10)

    print(f"\n生成的因子表达式:")
    for i, expr in enumerate(factor_exprs, 1):
        print(f"  {i}. {expr}")

    # 计算因子值
    print("\n正在计算因子值...")
    feature_data = {f: df[f].values for f in features}
    calculator = FactorCalculator()

    for i, expr in enumerate(factor_exprs, 1):
        factor_values = calculator.calculate_factor_value(expr, feature_data)
        if factor_values is not None:
            print(f"  因子 {i}: 计算成功，均值={np.mean(factor_values):.4f}, "
                  f"标准差={np.std(factor_values):.4f}")

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    tushare_example()
