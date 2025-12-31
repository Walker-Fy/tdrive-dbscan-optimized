"""
性能分析可视化
算法性能比较和优化结果可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
from datetime import datetime


class PerformanceVisualizer:
    """性能可视化器"""

    def __init__(self, figsize: Tuple[int, int] = (10, 8),
                 style: str = 'seaborn-v0_8'):
        """
        初始化可视化器

        Args:
            figsize: 图形大小
            style: matplotlib样式
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)

    def plot_execution_time_comparison(self, results: Dict[str, Dict[str, Any]],
                                       title: str = "执行时间比较",
                                       save_path: Optional[str] = None,
                                       log_scale: bool = False) -> plt.Figure:
        """
        绘制执行时间比较图

        Args:
            results: 结果字典 {实现名称: 性能数据}
            title: 图表标题
            save_path: 保存路径
            log_scale: 是否使用对数坐标轴

        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 准备数据
        implementations = []
        execution_times = []
        speedups = []
        memory_usages = []
        accuracies = []

        for name, data in results.items():
            if 'execution_time' in data:
                implementations.append(name)
                execution_times.append(data['execution_time'])
                speedups.append(data.get('speedup', 1.0))
                memory_usages.append(data.get('memory_usage_mb', 0))
                accuracies.append(data.get('accuracy', 0))

        if not implementations:
            print("没有性能数据可绘制")
            return fig

        # 1. 执行时间柱状图
        ax1 = axes[0, 0]
        bars1 = ax1.bar(implementations, execution_times,
                        color=plt.cm.Set3(np.linspace(0, 1, len(implementations))))
        ax1.set_title('执行时间比较', fontsize=12, fontweight='bold')
        ax1.set_ylabel('时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)

        if log_scale:
            ax1.set_yscale('log')

        # 在柱状图上添加数值
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height * 1.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. 速度提升
        ax2 = axes[0, 1]
        if any(s > 1 for s in speedups):
            bars2 = ax2.bar(implementations, speedups,
                            color=plt.cm.Set2(np.linspace(0, 1, len(implementations))))
            ax2.set_title('速度提升（相对于基准）', fontsize=12, fontweight='bold')
            ax2.set_ylabel('速度提升倍数')
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax2.tick_params(axis='x', rotation=45)

        # 3. 内存使用
        ax3 = axes[1, 0]
        bars3 = ax3.bar(implementations, memory_usages,
                        color=plt.cm.Pastel1(np.linspace(0, 1, len(implementations))))
        ax3.set_title('内存使用比较', fontsize=12, fontweight='bold')
        ax3.set_ylabel('内存 (MB)')
        ax3.tick_params(axis='x', rotation=45)

        if log_scale:
            ax3.set_yscale('log')

        # 4. 准确性比较
        ax4 = axes[1, 1]
        if any(a > 0 for a in accuracies):
            bars4 = ax4.bar(implementations, accuracies,
                            color=plt.cm.Paired(np.linspace(0, 1, len(implementations))))
            ax4.set_title('准确性比较', fontsize=12, fontweight='bold')
            ax4.set_ylabel('准确性')
            ax4.set_ylim(0, 1.1)
            ax4.tick_params(axis='x', rotation=45)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能比较图已保存到: {save_path}")

        return fig

    def plot_scalability_analysis(self, scalability_data: Dict[str, Any],
                                  title: str = "可扩展性分析",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制可扩展性分析图

        Args:
            scalability_data: 可扩展性数据
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        sizes = scalability_data.get('sizes', [])
        execution_times = scalability_data.get('execution_times', [])
        memory_usages = scalability_data.get('memory_usages', [])

        if len(sizes) != len(execution_times) or len(sizes) < 2:
            print("可扩展性数据不足")
            return fig

        # 转换为numpy数组
        sizes_array = np.array(sizes[:len(execution_times)])
        times_array = np.array(execution_times)

        if len(memory_usages) == len(sizes):
            memory_array = np.array(memory_usages)
        else:
            memory_array = None

        # 1. 执行时间 vs 数据大小
        ax1 = axes[0, 0]
        ax1.plot(sizes_array, times_array, 'bo-', linewidth=2, markersize=8, label='实际时间')
        ax1.set_xlabel('数据大小')
        ax1.set_ylabel('执行时间 (秒)')
        ax1.set_title('执行时间 vs 数据大小', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 添加复杂度拟合线
        if 'fits' in scalability_data:
            fits = scalability_data['fits']
            x_fit = np.linspace(min(sizes_array), max(sizes_array), 100)

            for model, fit_data in fits.items():
                coeff = fit_data['coeff']
                r2 = fit_data['r2']

                if model == 'linear':
                    y_fit = np.polyval(coeff, x_fit)
                    label = f'O(n), R²={r2:.3f}'
                    ax1.plot(x_fit, y_fit, 'r--', alpha=0.5, label=label)
                elif model == 'quadratic':
                    y_fit = np.polyval(coeff, x_fit)
                    label = f'O(n²), R²={r2:.3f}'
                    ax1.plot(x_fit, y_fit, 'g--', alpha=0.5, label=label)
                elif model == 'nlogn':
                    x_log_fit = x_fit * np.log2(x_fit + 1)
                    y_fit = np.polyval(coeff, x_log_fit)
                    label = f'O(n log n), R²={r2:.3f}'
                    ax1.plot(x_fit, y_fit, 'y--', alpha=0.5, label=label)

            ax1.legend()

        # 2. 内存使用 vs 数据大小
        ax2 = axes[0, 1]
        if memory_array is not None:
            ax2.plot(sizes_array, memory_array, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('数据大小')
            ax2.set_ylabel('内存使用 (MB)')
            ax2.set_title('内存使用 vs 数据大小', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # 3. 对数-对数图（用于分析时间复杂度）
        ax3 = axes[1, 0]
        ax3.loglog(sizes_array, times_array, 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('数据大小 (log)')
        ax3.set_ylabel('执行时间 (log)')
        ax3.set_title('对数-对数图：时间复杂度分析', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')

        # 添加参考线
        x_ref = np.array([min(sizes_array), max(sizes_array)])

        # O(n) 参考线
        y_ref_n = times_array[0] * x_ref / sizes_array[0]
        ax3.loglog(x_ref, y_ref_n, 'r--', alpha=0.5, label='O(n)')

        # O(n²) 参考线
        y_ref_n2 = times_array[0] * (x_ref / sizes_array[0]) ** 2
        ax3.loglog(x_ref, y_ref_n2, 'g--', alpha=0.5, label='O(n²)')

        ax3.legend()

        # 4. 加速比（如果有多核数据）
        ax4 = axes[1, 1]
        if 'speedups' in scalability_data:
            speedups = scalability_data['speedups']
            cores = scalability_data.get('cores', range(1, len(speedups) + 1))

            ax4.plot(cores, speedups, 'ro-', linewidth=2, markersize=8, label='实际加速比')
            ax4.plot(cores, cores, 'b--', linewidth=2, label='理想加速比')
            ax4.set_xlabel('核心数')
            ax4.set_ylabel('加速比')
            ax4.set_title('并行加速比分析', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            # 计算并行效率
            efficiency = [s / c for s, c in zip(speedups, cores)]
            ax4_twin = ax4.twinx()
            ax4_twin.plot(cores, efficiency, 'g:', linewidth=2, label='并行效率')
            ax4_twin.set_ylabel('并行效率')
            ax4_twin.legend(loc='upper right')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可扩展性分析图已保存到: {save_path}")

        return fig

    def plot_memory_usage_analysis(self, memory_data: Dict[str, Any],
                                   title: str = "内存使用分析",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制内存使用分析图

        Args:
            memory_data: 内存数据
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 内存使用随时间变化
        ax1 = axes[0, 0]
        if 'timestamps' in memory_data and 'memory_usage' in memory_data:
            timestamps = memory_data['timestamps']
            memory_usage = memory_data['memory_usage']

            ax1.plot(timestamps, memory_usage, 'b-', linewidth=2)
            ax1.fill_between(timestamps, 0, memory_usage, alpha=0.3)
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('内存使用 (MB)')
            ax1.set_title('内存使用随时间变化', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 标记峰值
            peak_idx = np.argmax(memory_usage)
            ax1.plot(timestamps[peak_idx], memory_usage[peak_idx], 'ro', markersize=10)
            ax1.annotate(f'峰值: {memory_usage[peak_idx]:.1f} MB',
                         xy=(timestamps[peak_idx], memory_usage[peak_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 2. 内存分配热点
        ax2 = axes[0, 1]
        if 'memory_hotspots' in memory_data and memory_data['memory_hotspots']:
            hotspots = memory_data['memory_hotspots']
            functions = [h.get('function', f'热点{i}')[:30] for i, h in enumerate(hotspots)]
            sizes = [h.get('size_mb', 0) for h in hotspots]

            # 只显示前10个热点
            if len(functions) > 10:
                functions = functions[:10]
                sizes = sizes[:10]

            bars = ax2.barh(range(len(functions)), sizes,
                            color=plt.cm.Reds(np.linspace(0.3, 0.9, len(functions))))
            ax2.set_yticks(range(len(functions)))
            ax2.set_yticklabels(functions)
            ax2.set_xlabel('内存使用 (MB)')
            ax2.set_title('内存分配热点', fontsize=12, fontweight='bold')
            ax2.invert_yaxis()  # 最大的在顶部

        # 3. 内存增长模式
        ax3 = axes[1, 0]
        if 'memory_growth' in memory_data:
            growth_data = memory_data['memory_growth']
            phases = list(growth_data.keys())
            growth_rates = [growth_data[p] for p in phases]

            bars = ax3.bar(phases, growth_rates,
                           color=plt.cm.Blues(np.linspace(0.3, 0.9, len(phases))))
            ax3.set_xlabel('阶段')
            ax3.set_ylabel('内存增长率 (MB/秒)')
            ax3.set_title('内存增长模式分析', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)

        # 4. 对象类型内存使用
        ax4 = axes[1, 1]
        if 'object_types' in memory_data:
            object_data = memory_data['object_types']
            obj_types = list(object_data.keys())
            obj_sizes = [object_data[t] for t in obj_types]

            # 创建饼图
            wedges, texts, autotexts = ax4.pie(obj_sizes, labels=obj_types, autopct='%1.1f%%',
                                               startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(obj_types))))
            ax4.set_title('对象类型内存分布', fontsize=12, fontweight='bold')

            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"内存分析图已保存到: {save_path}")

        return fig

    def plot_parallel_efficiency(self, parallel_data: Dict[str, Any],
                                 title: str = "并行效率分析",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制并行效率分析图

        Args:
            parallel_data: 并行数据
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if 'n_workers' not in parallel_data or 'speedups' not in parallel_data:
            print("缺少并行数据")
            return fig

        n_workers = parallel_data['n_workers']
        speedups = parallel_data['speedups']

        # 计算并行效率
        efficiency = [s / n for s, n in zip(speedups, n_workers)]

        # 绘制加速比
        ax.plot(n_workers, speedups, 'bo-', linewidth=2, markersize=8, label='实际加速比')
        ax.plot(n_workers, n_workers, 'r--', linewidth=2, label='理想加速比')
        ax.set_xlabel('工作进程数')
        ax.set_ylabel('加速比', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)

        # 绘制并行效率（次坐标轴）
        ax2 = ax.twinx()
        ax2.plot(n_workers, efficiency, 'g^--', linewidth=2, markersize=8, label='并行效率')
        ax2.set_ylabel('并行效率', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0, 1.1)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax.set_title(title, fontsize=14, fontweight='bold')

        # 添加统计信息
        max_speedup = max(speedups)
        max_efficiency = max(efficiency)
        optimal_workers = n_workers[efficiency.index(max_efficiency)]

        stats_text = f'最大加速比: {max_speedup:.2f}\n最大并行效率: {max_efficiency:.2%}\n最优工作进程数: {optimal_workers}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"并行效率分析图已保存到: {save_path}")

        return fig

    def generate_comprehensive_report(self, performance_data: Dict[str, Any],
                                      output_dir: str = "./results") -> None:
        """
        生成综合性能报告

        Args:
            performance_data: 性能数据
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 执行时间比较图
        if 'implementations' in performance_data:
            self.plot_execution_time_comparison(
                performance_data['implementations'],
                title="算法实现性能比较",
                save_path=str(output_path / f"execution_time_comparison_{timestamp}.png")
            )

        # 2. 可扩展性分析图
        if 'scalability' in performance_data:
            self.plot_scalability_analysis(
                performance_data['scalability'],
                title="算法可扩展性分析",
                save_path=str(output_path / f"scalability_analysis_{timestamp}.png")
            )

        # 3. 内存使用分析图
        if 'memory' in performance_data:
            self.plot_memory_usage_analysis(
                performance_data['memory'],
                title="内存使用分析",
                save_path=str(output_path / f"memory_analysis_{timestamp}.png")
            )

        # 4. 并行效率分析图
        if 'parallel' in performance_data:
            self.plot_parallel_efficiency(
                performance_data['parallel'],
                title="并行效率分析",
                save_path=str(output_path / f"parallel_efficiency_{timestamp}.png")
            )

        print(f"综合性能报告已生成到: {output_path}")


def plot_performance_comparison(results: Dict[str, Dict[str, Any]],
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    快速绘制性能比较图

    Args:
        results: 性能结果字典
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    visualizer = PerformanceVisualizer()
    return visualizer.plot_execution_time_comparison(results, save_path=save_path)


def plot_scalability_analysis(scalability_data: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    快速绘制可扩展性分析图

    Args:
        scalability_data: 可扩展性数据
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    visualizer = PerformanceVisualizer()
    return visualizer.plot_scalability_analysis(scalability_data, save_path=save_path)


def plot_memory_usage(memory_data: Dict[str, Any],
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    快速绘制内存使用分析图

    Args:
        memory_data: 内存数据
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    visualizer = PerformanceVisualizer()
    return visualizer.plot_memory_usage_analysis(memory_data, save_path=save_path)