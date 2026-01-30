"""Main entry point for AlphaGPT - Part 1: Imports and initialization"""

import warnings
import numpy as np
import pandas as pd
import logging
import os

from alphagpt.config import Config
from alphagpt.utils import setup_logger, set_seed, create_run_output_dir
from alphagpt.data import DataLoader, FeatureEngineer
from alphagpt.factor import FactorGenerator, FactorCalculator
from alphagpt.backtest import BacktestEngine
from alphagpt.visualization import ResultPlotter

warnings.filterwarnings('ignore')


def load_and_merge_multiple_stocks(config, logger):
    """
    加载并合并多个股票的数据

    Args:
        config: 配置对象
        logger: 日志对象

    Returns:
        合并后的DataFrame
    """
    print(f"\n正在加载 {len(config.codes)} 个标的的数据...")

    # 批量加载多个股票数据
    stock_data_dict = DataLoader.get_multiple_price_data(
        codes=config.codes,
        start=config.start_date,
        end=config.test_end
    )

    # 合并所有股票数据
    all_dfs = []
    for code, df in stock_data_dict.items():
        df = df.copy()
        df['code'] = code  # 添加股票代码列
        all_dfs.append(df)

    df_merged = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"合并后数据总行数: {len(df_merged)}")

    return df_merged


def main():
    """Main function to run AlphaGPT system"""
    print("=" * 60)
    print("AlphaGPT 量化因子生成与回测系统")
    print("=" * 60)

    # Load configuration from YAML file
    config = Config.from_yaml('config.yaml')

    # Setup logger
    logger = setup_logger(__name__, config.log_level, config.log_file)

    # 记录配置信息
    logger.info("=" * 60)
    logger.info("AlphaGPT 量化因子生成与回测系统启动")
    logger.info("=" * 60)
    logger.info(f"配置信息:")
    logger.info(f"  - 股票代码: {config.codes}")
    logger.info(f"  - 数据日期范围: {config.start_date} ~ {config.test_end}")
    logger.info(f"  - 训练集截止: {config.end_date}")
    logger.info(f"  - 因子生成数量: {config.num_factors}")
    logger.info(f"  - 最大序列长度: {config.max_seq_len}")
    logger.info(f"  - 最小操作数: {config.min_op_count}")
    logger.info(f"  - API模型: {config.model}")
    logger.info(f"  - 随机种子: {config.random_seed}")

    # Set random seed
    set_seed(config.random_seed)

    # Create output directory with subfolder structure: results/YYYY-MM-DD/stock_codes/
    output_dir = create_run_output_dir(
        base_dir="results",
        codes=config.codes
    )
    logger.info(f"输出目录: {output_dir}")

    # Load data - 支持多标的
    logger.info("开始加载数据...")
    df_all = load_and_merge_multiple_stocks(config, logger)

    if df_all is None or len(df_all) == 0:
        logger.error("数据加载失败，程序退出")
        print("数据加载失败，程序退出")
        return

    # Split data
    split_date = pd.to_datetime(config.end_date).date()
    train_df_raw = df_all[df_all.trade_date <= split_date].copy()
    test_df_raw = df_all[df_all.trade_date > split_date].copy()
    logger.info(f"数据划分完成: 训练集 {len(train_df_raw)} 行, 测试集 {len(test_df_raw)} 行")

    # Feature engineering
    logger.info("开始特征工程...")
    print("进行特征工程...")
    train_df, features = FeatureEngineer.create_features(train_df_raw)
    test_df, _ = FeatureEngineer.create_features(test_df_raw)

    features = [f for f in features if f in train_df.columns and f in test_df.columns]
    train_df = train_df[['trade_date', 'close'] + features].copy()
    test_df = test_df[['trade_date', 'close'] + features].copy()

    logger.info(f"特征工程完成: 生成 {len(features)} 个特征")
    logger.info(f"训练集: {len(train_df)} 样本 ({train_df['trade_date'].min()} ~ {train_df['trade_date'].max()})")
    logger.info(f"测试集: {len(test_df)} 样本 ({test_df['trade_date'].min()} ~ {test_df['trade_date'].max()})")

    print(f"训练集: {len(train_df)} 个样本 ({train_df['trade_date'].min()} 到 {train_df['trade_date'].max()})")
    print(f"测试集: {len(test_df)} 个样本 ({test_df['trade_date'].min()} 到 {test_df['trade_date'].max()})")
    print(f"特征: {', '.join(features)}")

    # Calculate target returns
    train_returns = np.zeros(len(train_df))
    train_returns[:-1] = train_df['close'].values[1:] / train_df['close'].values[:-1] - 1
    train_returns = np.clip(train_returns, -0.1, 0.1)

    test_returns = np.zeros(len(test_df))
    test_returns[:-1] = test_df['close'].values[1:] / test_df['close'].values[:-1] - 1
    test_returns = np.clip(test_returns, -0.1, 0.1)

    # Initialize factor generator
    print("\n初始化Gemini因子生成器...")
    logger.info("初始化因子生成器...")
    factor_generator = FactorGenerator(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        proxy=config.gemini_proxy
    )

    # Generate factors
    print(f"\n生成量化因子...")
    logger.info(f"开始生成 {config.num_factors} 个因子...")
    factor_exprs = factor_generator.generate_factors_with_gemini(
        features,
        max_seq_len=config.max_seq_len,
        min_op_count=config.min_op_count,
        num_factors=config.num_factors
    )

    if not factor_exprs:
        print("未能生成有效因子，使用默认因子")
        logger.warning("API未能生成有效因子，使用默认因子")
        factor_exprs = [
            "ret_norm + vol_chg_norm * trend_norm",
            "abs(ret5_norm) - vol_chg_norm",
            "mom5_norm * vol_chg_norm",
            "ret_norm / (trend_norm + 1e-6)",
            "ret_norm - vol_chg_norm"
        ]

    print(f"\n生成的因子表达式 ({len(factor_exprs)} 个):")
    logger.info(f"成功生成 {len(factor_exprs)} 个因子表达式:")
    for i, expr in enumerate(factor_exprs, 1):
        print(f"{i}. {expr}")
        logger.info(f"  Factor_{i}: {expr}")

    # Prepare feature data
    train_feature_data = {f: train_df[f].values for f in features}
    test_feature_data = {f: test_df[f].values for f in features}

    # Initialize calculator
    calculator = FactorCalculator()

    # Evaluate and backtest all factors
    print(f"\n评估和回测因子...")
    logger.info("开始评估和回测因子...")
    results = {}
    best_sharpe = -np.inf
    best_factor_name = None

    for i, expr in enumerate(factor_exprs, 1):
        factor_name = f"Factor_{i}"
        print(f"\n[{i}/{len(factor_exprs)}] 评估 {factor_name}")
        print(f"表达式: {expr}")
        logger.info(f"评估 {factor_name}: {expr}")

        # Evaluate on training set
        factor_values_train = calculator.calculate_factor_value(expr, train_feature_data)

        if factor_values_train is not None and len(factor_values_train) == len(train_df):
            train_sharpe = calculator.calculate_sharpe(factor_values_train, train_returns)
            print(f"训练集夏普比率: {train_sharpe:.3f}")
            logger.info(f"  训练集夏普比率: {train_sharpe:.3f}")

            # Calculate factor values on test set
            factor_values_test = calculator.calculate_factor_value(expr, test_feature_data)

            if factor_values_test is not None and len(factor_values_test) == len(test_df):
                # Backtest strategy
                result = BacktestEngine.backtest_strategy(
                    factor_values_test, test_df, test_returns, factor_name
                )

                if result is not None:
                    results[factor_name] = result
                    print(f"测试集夏普比率: {result['sharpe']:.3f}")
                    logger.info(f"  测试集夏普比率: {result['sharpe']:.3f}, 年化收益: {result['annual_return']:.2%}, 最大回撤: {result['max_drawdown']:.2%}")

                    # Update best factor
                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_factor_name = factor_name
                else:
                    print(f"测试集回测失败")
                    logger.warning(f"  {factor_name} 测试集回测失败")
            else:
                print(f"测试集因子计算失败")
                logger.warning(f"  {factor_name} 测试集因子计算失败")
        else:
            print(f"训练集因子计算失败")
            logger.warning(f"  {factor_name} 训练集因子计算失败")

    # Output results summary
    print(f"\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("结果汇总")
    logger.info("=" * 60)

    if best_factor_name and best_factor_name in results:
        best_result = results[best_factor_name]
        # 安全解析因子索引
        try:
            best_expr_idx = int(best_factor_name.split('_')[1]) - 1
            best_expr = factor_exprs[best_expr_idx] if 0 <= best_expr_idx < len(factor_exprs) else "N/A"
        except (IndexError, ValueError):
            best_expr = "N/A"

        print(f"🏆 最佳因子: {best_factor_name}")
        print(f"   表达式: {best_expr}")
        print(f"\n📊 测试集绩效:")
        print(f"   夏普比率: {best_result['sharpe']:.3f}")
        print(f"   年化收益率: {best_result['annual_return']:.2%}")
        print(f"   年化波动率: {best_result['annual_vol']:.2%}")
        print(f"   最大回撤: {best_result['max_drawdown']:.2%}")
        print(f"   总收益率: {best_result['total_return']:.2%}")
        print(f"   信息比率: {best_result['info_ratio']:.2f}")
        print(f"   胜率: {best_result['win_rate']:.1f}%")
        print(f"   卡玛比率: {best_result['calmar']:.2f}")

        # 记录最佳因子到日志
        logger.info(f"最佳因子: {best_factor_name}")
        logger.info(f"表达式: {best_expr}")
        logger.info(f"测试集绩效:")
        logger.info(f"  夏普比率: {best_result['sharpe']:.3f}")
        logger.info(f"  年化收益率: {best_result['annual_return']:.2%}")
        logger.info(f"  年化波动率: {best_result['annual_vol']:.2%}")
        logger.info(f"  最大回撤: {best_result['max_drawdown']:.2%}")
        logger.info(f"  总收益率: {best_result['total_return']:.2%}")
        logger.info(f"  信息比率: {best_result['info_ratio']:.2f}")
        logger.info(f"  胜率: {best_result['win_rate']:.1f}%")
        logger.info(f"  卡玛比率: {best_result['calmar']:.2f}")
        logger.info(f"  信息系数(IC): {best_result.get('ic', np.nan):.3f}")

        # Calculate factor values for the best factor (for rolling metrics)
        best_expr = factor_exprs[int(best_factor_name.split('_')[1]) - 1]
        best_factor_values = calculator.calculate_factor_value(best_expr, test_feature_data)

        # Plot results with enhanced visualization (using simpler filename)
        ResultPlotter.plot_results(
            results,
            test_df,
            best_factor_name,
            "performance",  # Simplified filename - directory already identifies the run
            output_dir,
            factor_values=best_factor_values,
            test_returns=test_returns
        )

        # Save detailed results (using simpler filename)
        try:
            summary_data = []
            for factor_name, result in results.items():
                if result is not None:
                    expr_idx = int(factor_name.split('_')[1]) - 1
                    expr = factor_exprs[expr_idx] if expr_idx < len(factor_exprs) else "N/A"
                    ic = result.get('ic', np.nan)
                    summary_data.append({
                        'factor_name': factor_name,
                        'expression': expr,
                        'sharpe': result['sharpe'],
                        'ic': ic,
                        'annual_return': result['annual_return'],
                        'annual_vol': result['annual_vol'],
                        'max_drawdown': result['max_drawdown'],
                        'total_return': result['total_return'],
                        'info_ratio': result['info_ratio'],
                        'win_rate': result['win_rate'],
                        'calmar': result['calmar']
                    })

            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, 'summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n📁 详细结果已保存至: {summary_path}")
            logger.info(f"详细结果已保存至: {summary_path}")

        except Exception as e:
            print(f"保存结果文件失败: {e}")
            logger.error(f"保存结果文件失败: {e}")
    else:
        print("未找到有效的最佳因子")
        logger.warning("未找到有效的最佳因子")

    print(f"\n" + "=" * 60)
    print("程序执行完毕！")
    print("=" * 60)
    logger.info("程序执行完毕")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        logging.error(f"程序异常退出: {type(e).__name__}: {e}")
        print(f"\n程序异常退出: {e}")
        raise
