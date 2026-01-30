"""
AlphaGPT 使用示例
演示如何使用 qlib 获取股票数据并进行因子分析
"""

from alphagpt.config import Config
from alphagpt.utils import setup_logger, set_seed
from alphagpt.data import DataLoader, FeatureEngineer
from alphagpt.factor import FactorGenerator, FactorCalculator
import numpy as np


def example():
    """qlib 数据获取示例"""

    print("=" * 60)
    print("AlphaGPT 使用示例")
    print("=" * 60)

    # 从 YAML 配置文件加载配置
    config = Config.from_yaml('config.yaml')
    print(config)

    # 设置日志和随机种子
    logger = setup_logger(__name__)
    set_seed(42)

    # 加载数据
    print(f"\n正在加载数据: {config.codes[0]}")
    print(f"日期范围: {config.start_date} 到 {config.end_date}")

    df = DataLoader.get_price_data(
        code=config.codes[0],
        start=config.start_date,
        end=config.end_date
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
    generator = FactorGenerator(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        proxy=config.gemini_proxy
    )
    # factor_exprs = generator.generate_factors_local(features, num_factors=10)
    factor_exprs = generator.generate_factors_with_gemini(
        features,
        max_seq_len=config.max_seq_len,
        min_op_count=config.min_op_count,
        num_factors=config.num_factors
    )

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
    example()
