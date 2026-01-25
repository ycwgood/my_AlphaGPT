"""Result plotting module - Part 1: Imports and class definition"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import platform
import os

logger = logging.getLogger(__name__)

# Configure Chinese font for matplotlib
def _configure_chinese_font():
    """Configure matplotlib to support Chinese characters"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'SimHei']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


class ResultPlotter:
    """Plot backtest results and performance metrics"""

    @staticmethod
    def plot_results(
        results: Dict,
        test_df: pd.DataFrame,
        best_factor_name: str,
        index_code: str = "performance",
        output_dir: str = "results",
        factor_values: np.ndarray = None,
        test_returns: np.ndarray = None
    ) -> None:
        """
        Plot backtest results with enhanced 2x3 layout

        Args:
            results: Dictionary of backtest results
            test_df: Test DataFrame
            best_factor_name: Name of best performing factor
            index_code: Base name for output file (default: "performance")
            output_dir: Output directory for saving plots
            factor_values: Factor values for rolling metrics (optional)
            test_returns: Test returns for rolling metrics (optional)
        """
        if not results:
            print("无有效结果可绘制")
            return

        # Configure Chinese font
        _configure_chinese_font()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # First row: Net value, Drawdown, Rolling metrics
        ResultPlotter._plot_net_value(axes[0, 0], results, test_df, best_factor_name)

        if best_factor_name in results and results[best_factor_name] is not None:
            ResultPlotter._plot_drawdown(axes[0, 1], results[best_factor_name], test_df, best_factor_name)

            # Plot rolling metrics if factor values and returns provided
            if factor_values is not None and test_returns is not None:
                ResultPlotter._plot_rolling_metrics(
                    axes[0, 2],
                    factor_values,
                    test_returns,
                    test_df
                )
            else:
                axes[0, 2].text(0.5, 0.5, '需要提供因子值和收益率\n以绘制滚动指标',
                                ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('滚动指标')

        # Second row: Monthly returns, Positions, Metrics table
        ResultPlotter._plot_monthly_returns(axes[1, 0], results, test_df, best_factor_name)
        ResultPlotter._plot_positions(axes[1, 1], results, test_df, best_factor_name)
        ResultPlotter._plot_metrics_table(axes[1, 2], results, best_factor_name)

        plt.tight_layout()

        # Save figure
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Simplified filename - directory already identifies the run
            save_path = os.path.join(output_dir, f'{index_code}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"绩效图已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存图片失败: {e}")

        plt.show()

    @staticmethod
    def _plot_net_value(ax, results: Dict, test_df: pd.DataFrame, best_factor_name: str) -> None:
        """Plot net value curves"""
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for idx, (factor_name, result) in enumerate(results.items()):
            if result is not None:
                ax.plot(
                    test_df['trade_date'].values[:len(result['cumulative'])],
                    result['cumulative'],
                    label=f"{factor_name} (Sharpe: {result['sharpe']:.2f})",
                    linewidth=2 if factor_name == best_factor_name else 1,
                    alpha=0.8 if factor_name == best_factor_name else 0.5,
                    color=colors[idx]
                )

        # Add benchmark
        if best_factor_name in results and results[best_factor_name] is not None:
            ax.plot(
                test_df['trade_date'].values[:len(results[best_factor_name]['benchmark_cumulative'])],
                results[best_factor_name]['benchmark_cumulative'],
                label='基准持有',
                linewidth=1.5,
                color='gray',
                alpha=0.7,
                linestyle='--'
            )

        ax.set_title(f'因子策略净值曲线对比 | 最佳: {best_factor_name}')
        ax.set_ylabel('累计净值')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    @staticmethod
    def _plot_monthly_returns(ax, results: Dict, test_df: pd.DataFrame, best_factor_name: str) -> None:
        """Plot monthly returns for best factor"""
        if best_factor_name in results and results[best_factor_name] is not None:
            best_result = results[best_factor_name]
            dates = pd.to_datetime(test_df['trade_date'].values[:len(best_result['strategy_returns'])])
            monthly_returns = pd.Series(best_result['strategy_returns'], index=dates).resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            )

            colors_bar = ['red' if r < 0 else 'green' for r in monthly_returns]
            ax.bar(range(len(monthly_returns)), monthly_returns.values * 100, color=colors_bar, alpha=0.7)
            ax.set_title(f'{best_factor_name} 月度收益率 (%)')
            ax.set_ylabel('收益率 %')
            ax.set_xticks(range(len(monthly_returns)))
            ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly_returns.index], rotation=45)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')

    @staticmethod
    def _plot_positions(ax, results: Dict, test_df: pd.DataFrame, best_factor_name: str) -> None:
        """Plot position changes for best factor"""
        if best_factor_name in results and results[best_factor_name] is not None:
            best_result = results[best_factor_name]
            ax.plot(
                test_df['trade_date'].values[:len(best_result['positions']) - 1],
                best_result['positions'][:-1],
                label='仓位',
                linewidth=1,
                color='purple'
            )
            ax.set_title('仓位变化')
            ax.set_xlabel('日期')
            ax.set_ylabel('仓位')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    @staticmethod
    def _plot_drawdown(ax, result: Dict, test_df: pd.DataFrame, factor_name: str) -> None:
        """Plot drawdown curve"""
        if 'drawdown' not in result or result['drawdown'] is None:
            ax.text(0.5, 0.5, '暂无回撤数据', ha='center', va='center', transform=ax.transAxes)
            return

        drawdown = result['drawdown']
        dates = test_df['trade_date'].values[:len(drawdown)]

        # Fill drawdown area
        ax.fill_between(
            dates,
            drawdown * 100,
            0,
            color='red',
            alpha=0.3,
            label='回撤'
        )
        ax.plot(dates, drawdown * 100, color='darkred', linewidth=1)

        # Add max drawdown line
        max_dd = result.get('max_drawdown', 0)
        ax.axhline(y=max_dd * 100, color='darkred', linestyle='--', alpha=0.7,
                   label=f'最大回撤: {max_dd:.2%}')

        ax.set_title(f'{factor_name} 回撤曲线 (%)')
        ax.set_ylabel('回撤 %')
        ax.set_xlabel('日期')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        ax.tick_params(axis='x', rotation=45)

    @staticmethod
    def _plot_rolling_metrics(
        ax,
        factor_values: np.ndarray,
        returns: np.ndarray,
        test_df: pd.DataFrame,
        window: int = 60
    ) -> None:
        """Plot rolling IC and Sharpe ratio"""
        from alphagpt.backtest.engine import BacktestEngine

        # Calculate rolling IC
        rolling_ic = BacktestEngine.calculate_rolling_ic(
            factor_values[:len(returns)],
            returns,
            window=window
        )

        # Calculate rolling Sharpe
        rolling_sharpe = []
        for i in range(window, len(returns)):
            ret_window = returns[i-window:i]
            if len(ret_window) > 0 and np.std(ret_window) > 1e-6:
                sharpe = np.mean(ret_window) / np.std(ret_window) * np.sqrt(252)
                rolling_sharpe.append(sharpe)
            else:
                rolling_sharpe.append(0)

        # Plot on dual y-axis
        dates = test_df['trade_date'].values[window:window+len(rolling_ic)]

        ax2 = ax.twinx()

        line1 = ax.plot(dates, rolling_ic, color='blue', label='滚动IC',
                        linewidth=1.5, alpha=0.8)
        line2 = ax2.plot(dates[:len(rolling_sharpe)], rolling_sharpe,
                         color='orange', label='滚动夏普', linewidth=1.5,
                         alpha=0.8, linestyle='--')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        ax.set_ylabel('IC', color='blue')
        ax2.set_ylabel('夏普', color='orange')
        ax.set_title(f'滚动指标 ({window}日窗口)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)

    @staticmethod
    def _plot_metrics_table(ax, results: Dict, best_factor_name: str) -> None:
        """Plot performance metrics table"""
        ax.axis('tight')
        ax.axis('off')

        if results:
            perf_data = []
            for factor_name, result in results.items():
                if result is not None:
                    ic = result.get('ic', np.nan)
                    ic_str = f"{ic:.3f}" if not np.isnan(ic) else "N/A"
                    perf_data.append([
                        factor_name,
                        f"{result['sharpe']:.2f}",
                        ic_str,
                        f"{result['annual_return']:.2%}",
                        f"{result['max_drawdown']:.2%}",
                        f"{result['info_ratio']:.2f}",
                        f"{result['win_rate']:.1f}%"
                    ])

            if perf_data:
                table = ax.table(
                    cellText=perf_data,
                    colLabels=['因子', '夏普', 'IC', '年化收益', '最大回撤', '信息比率', '胜率'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
                )

                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.8)

                # Highlight best factor row
                for i, row in enumerate(perf_data):
                    if row[0] == best_factor_name:
                        for j in range(len(row)):
                            table[(i + 1, j)].set_facecolor('#90EE90')

                # Highlight header row
                for j in range(len(perf_data[0])):
                    table[(0, j)].set_facecolor('#E0E0E0')
                    table[(0, j)].set_text_props(weight='bold')

                ax.set_title('绩效指标汇总')

    @staticmethod
    def plot_factor_correlation(
        factor_dict: Dict[str, np.ndarray],
        output_dir: str = "results",
        filename: str = "factor_correlation.png"
    ) -> None:
        """
        Plot factor correlation heatmap (optional feature)

        Args:
            factor_dict: Dictionary of factor name to factor values
            output_dir: Output directory for saving plots
            filename: Output filename
        """
        try:
            # Import seaborn for better heatmap visualization
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn未安装，使用matplotlib绘制热力图")
            sns = None

        _configure_chinese_font()

        # Create correlation DataFrame
        corr_df = pd.DataFrame(factor_dict).corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        if sns is not None:
            # Use seaborn for better styling
            sns.heatmap(
                corr_df,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': '相关系数'},
                ax=ax
            )
        else:
            # Fallback to matplotlib
            im = ax.imshow(corr_df.values, cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_df.columns)))
            ax.set_yticks(range(len(corr_df.columns)))
            ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_df.columns)

            # Add text annotations
            for i in range(len(corr_df)):
                for j in range(len(corr_df.columns)):
                    text = ax.text(j, i, f'{corr_df.values[i, j]:.2f}',
                                   ha='center', va='center', color='black')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('相关系数')

        ax.set_title('因子相关性热力图', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        try:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"因子相关性图已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存相关性图失败: {e}")

        plt.show()
