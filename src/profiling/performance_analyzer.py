"""
综合性能分析器
比较不同实现并提供优化建议
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

from .memory_profiler import MemoryProfiler
from .time_profiler import TimeProfiler


@dataclass
class ImplementationResult:
    """实现结果"""
    name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    accuracy: Optional[float] = None
    n_clusters: Optional[int] = None
    n_noise: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    detailed_stats: Optional[Dict[str, Any]] = None


class PerformanceAnalyzer:
    """综合性能分析器"""

    def __init__(self, output_dir: str = "./results"):
        """
        初始化性能分析器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, ImplementationResult] = {}
        self.comparison_data: pd.DataFrame = pd.DataFrame()

    def benchmark_implementation(self, name: str,
                                 implementation_func: Callable,
                                 test_data: np.ndarray,
                                 reference_result: Optional[Dict] = None,
                                 **kwargs) -> ImplementationResult:
        """
        基准测试一个实现

        Args:
            name: 实现名称
            implementation_func: 实现函数
            test_data: 测试数据
            reference_result: 参考结果（用于精度评估）
            **kwargs: 传递给实现函数的参数

        Returns:
            实现结果
        """
        print(f"开始基准测试: {name}")

        # 创建性能分析器
        time_profiler = TimeProfiler(enable_profiling=True)
        memory_profiler = MemoryProfiler(track_detailed=True)

        try:
            # 开始内存分析
            memory_profiler.start()

            # 分析函数性能
            memory_before = memory_profiler.take_snapshot("开始前")

            # 执行函数
            start_time = time.time()
            result, time_analysis = time_profiler.profile_function(
                implementation_func, test_data, **kwargs
            )
            execution_time = time.time() - start_time

            # 内存分析
            memory_after = memory_profiler.take_snapshot("结束后")
            memory_usage = memory_after.memory_usage_mb - memory_before.memory_usage_mb

            # 停止内存分析
            memory_profiler.stop()

            # 分析内存模式
            memory_patterns = memory_profiler.analyze_memory_patterns()

            # 计算准确性（如果有参考结果）
            accuracy = None
            if reference_result is not None and hasattr(result, 'labels_'):
                accuracy = self._calculate_accuracy(result.labels_, reference_result.get('labels', []))

            # 提取聚类结果
            n_clusters = None
            n_noise = None
            if hasattr(result, 'labels_'):
                unique_labels = np.unique(result.labels_)
                n_clusters = len([l for l in unique_labels if l != 0])
                n_noise = np.sum(result.labels_ == 0)

            # 创建结果对象
            impl_result = ImplementationResult(
                name=name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=memory_after.peak_memory_mb,
                accuracy=accuracy,
                n_clusters=n_clusters,
                n_noise=n_noise,
                success=True,
                detailed_stats={
                    'time_analysis': time_analysis,
                    'memory_patterns': memory_patterns,
                    'time_bottlenecks': time_profiler.analyze_performance_bottlenecks(),
                    'memory_bottlenecks': memory_profiler.find_memory_bottlenecks(),
                    'time_suggestions': time_profiler.suggest_optimizations(),
                    'memory_suggestions': memory_profiler.suggest_memory_optimizations()
                }
            )

            print(f"完成基准测试: {name}")
            print(f"  执行时间: {execution_time:.4f} 秒")
            print(f"  内存使用: {memory_usage:.2f} MB")
            print(f"  峰值内存: {memory_after.peak_memory_mb:.2f} MB")
            if accuracy is not None:
                print(f"  准确性: {accuracy:.4f}")
            if n_clusters is not None:
                print(f"  聚类数: {n_clusters}, 噪声点: {n_noise}")

        except Exception as e:
            print(f"基准测试 {name} 失败: {e}")

            # 创建失败结果
            impl_result = ImplementationResult(
                name=name,
                execution_time=0,
                memory_usage_mb=0,
                peak_memory_mb=0,
                success=False,
                error=str(e)
            )

        # 保存结果
        self.results[name] = impl_result

        return impl_result

    def _calculate_accuracy(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """
        计算两个聚类结果的相似度

        Args:
            labels1: 第一个聚类标签
            labels2: 第二个聚类标签

        Returns:
            相似度分数（0-1）
        """
        if len(labels1) != len(labels2):
            return 0.0

        # 使用调整兰德指数(ARI)的简化版本
        from sklearn.metrics import adjusted_rand_score
        try:
            return adjusted_rand_score(labels1, labels2)
        except:
            # 如果计算失败，使用简单的匹配率
            matches = np.sum(labels1 == labels2)
            return matches / len(labels1)

    def compare_implementations(self, implementations: Dict[str, Callable],
                                test_data: np.ndarray,
                                test_name: str = "benchmark",
                                **kwargs) -> pd.DataFrame:
        """
        比较多个实现

        Args:
            implementations: 实现字典 {名称: 函数}
            test_data: 测试数据
            test_name: 测试名称
            **kwargs: 通用参数

        Returns:
            比较结果DataFrame
        """
        print(f"开始比较 {len(implementations)} 个实现...")

        # 运行基准测试
        for name, impl_func in implementations.items():
            if name not in self.results:
                self.benchmark_implementation(
                    name, impl_func, test_data, **kwargs
                )

        # 创建比较DataFrame
        comparison_data = []

        for name, result in self.results.items():
            if result.success:
                row = {
                    'implementation': name,
                    'execution_time': result.execution_time,
                    'memory_usage_mb': result.memory_usage_mb,
                    'peak_memory_mb': result.peak_memory_mb,
                    'accuracy': result.accuracy or 0,
                    'n_clusters': result.n_clusters or 0,
                    'n_noise': result.n_noise or 0,
                    'success': True
                }

                # 添加详细统计
                if result.detailed_stats:
                    time_bottlenecks = result.detailed_stats.get('time_bottlenecks', [])
                    memory_bottlenecks = result.detailed_stats.get('memory_bottlenecks', [])

                    row['n_time_bottlenecks'] = len(time_bottlenecks)
                    row['n_memory_bottlenecks'] = len(memory_bottlenecks)

                    # 主要瓶颈
                    if time_bottlenecks:
                        row['main_time_bottleneck'] = time_bottlenecks[0]['function']

                    if memory_bottlenecks:
                        row['main_memory_bottleneck'] = memory_bottlenecks[0]['consumer']

                comparison_data.append(row)

        # 创建DataFrame
        self.comparison_data = pd.DataFrame(comparison_data)

        # 计算相对性能
        if len(self.comparison_data) > 1:
            best_time = self.comparison_data['execution_time'].min()
            best_memory = self.comparison_data['memory_usage_mb'].min()

            self.comparison_data['time_speedup'] = best_time / self.comparison_data['execution_time']
            self.comparison_data['memory_efficiency'] = best_memory / self.comparison_data['memory_usage_mb']

            # 综合评分
            self.comparison_data['score'] = (
                    0.5 * self.comparison_data['time_speedup'] +
                    0.3 * (1 / self.comparison_data['memory_usage_mb'].clip(lower=0.1)) +
                    0.2 * self.comparison_data['accuracy']
            )

        # 保存比较结果
        self._save_comparison_results(test_name)

        return self.comparison_data

    def analyze_scalability(self, implementation_func: Callable,
                            data_generator: Callable,
                            sizes: List[int],
                            **kwargs) -> Dict[str, Any]:
        """
        分析可扩展性

        Args:
            implementation_func: 实现函数
            data_generator: 数据生成函数
            sizes: 数据大小列表
            **kwargs: 函数参数

        Returns:
            可扩展性分析结果
        """
        scalability_results = {
            'sizes': sizes,
            'execution_times': [],
            'memory_usages': [],
            'complexities': []
        }

        for size in sizes:
            print(f"测试数据大小: {size}")

            # 生成测试数据
            test_data = data_generator(size)

            # 基准测试
            result = self.benchmark_implementation(
                f"scalability_{size}", implementation_func, test_data, **kwargs
            )

            if result.success:
                scalability_results['execution_times'].append(result.execution_time)
                scalability_results['memory_usages'].append(result.memory_usage_mb)

                # 估算时间复杂度
                if len(scalability_results['execution_times']) > 1:
                    # 简单的复杂度估计
                    n = size
                    t = result.execution_time
                    complexities = []

                    # O(n) 拟合
                    linear_fit = t / n
                    complexities.append(('O(n)', linear_fit))

                    # O(n log n) 拟合
                    nlogn_fit = t / (n * np.log2(n + 1))
                    complexities.append(('O(n log n)', nlogn_fit))

                    # O(n²) 拟合
                    n2_fit = t / (n ** 2)
                    complexities.append(('O(n²)', n2_fit))

                    scalability_results['complexities'].append(complexities)

        # 分析可扩展性模式
        if len(scalability_results['execution_times']) > 2:
            times = np.array(scalability_results['execution_times'])
            sizes_array = np.array(sizes[:len(times)])

            # 拟合增长曲线
            try:
                # 尝试不同复杂度模型的拟合
                fits = {}

                # 线性拟合
                linear_coeff = np.polyfit(sizes_array, times, 1)
                fits['linear'] = {'coeff': linear_coeff,
                                  'r2': self._calculate_r2(times, np.polyval(linear_coeff, sizes_array))}

                # 二次拟合
                quadratic_coeff = np.polyfit(sizes_array, times, 2)
                fits['quadratic'] = {'coeff': quadratic_coeff,
                                     'r2': self._calculate_r2(times, np.polyval(quadratic_coeff, sizes_array))}

                # n log n 拟合
                x_log = sizes_array * np.log2(sizes_array + 1)
                nlogn_coeff = np.polyfit(x_log, times, 1)
                fits['nlogn'] = {'coeff': nlogn_coeff, 'r2': self._calculate_r2(times, np.polyval(nlogn_coeff, x_log))}

                scalability_results['fits'] = fits

                # 确定最佳拟合
                best_fit = max(fits.items(), key=lambda x: x[1]['r2'])
                scalability_results['best_fit'] = {
                    'model': best_fit[0],
                    'r2': best_fit[1]['r2']
                }

            except Exception as e:
                print(f"拟合可扩展性曲线时出错: {e}")

        return scalability_results

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def generate_optimization_report(self, implementation_name: str) -> Dict[str, Any]:
        """
        生成优化报告

        Args:
            implementation_name: 实现名称

        Returns:
            优化报告
        """
        if implementation_name not in self.results:
            return {}

        result = self.results[implementation_name]

        if not result.success or not result.detailed_stats:
            return {}

        report = {
            'implementation': implementation_name,
            'summary': {
                'execution_time': result.execution_time,
                'memory_usage_mb': result.memory_usage_mb,
                'peak_memory_mb': result.peak_memory_mb,
                'accuracy': result.accuracy,
                'n_clusters': result.n_clusters,
                'n_noise': result.n_noise
            },
            'bottlenecks': {
                'time': result.detailed_stats.get('time_bottlenecks', []),
                'memory': result.detailed_stats.get('memory_bottlenecks', [])
            },
            'suggestions': {
                'time': result.detailed_stats.get('time_suggestions', []),
                'memory': result.detailed_stats.get('memory_suggestions', [])
            },
            'prioritized_optimizations': self._prioritize_optimizations(result)
        }

        return report

    def _prioritize_optimizations(self, result: ImplementationResult) -> List[Dict[str, Any]]:
        """
        优先级排序优化建议

        Args:
            result: 实现结果

        Returns:
            优先级排序的优化建议
        """
        optimizations = []

        if not result.detailed_stats:
            return optimizations

        # 时间瓶颈优化
        time_bottlenecks = result.detailed_stats.get('time_bottlenecks', [])
        for bottleneck in time_bottlenecks[:3]:  # 前3个时间瓶颈
            optimizations.append({
                'type': 'time',
                'priority': 'high' if bottleneck.get('time_ratio', 0) > 0.3 else 'medium',
                'description': f"优化函数 '{bottleneck.get('function', '')}'",
                'estimated_impact': f"减少 {bottleneck.get('time_ratio', 0):.1%} 的执行时间",
                'suggestions': [
                    "考虑向量化或并行化",
                    "使用更高效的数据结构",
                    "缓存计算结果"
                ]
            })

        # 内存瓶颈优化
        memory_bottlenecks = result.detailed_stats.get('memory_bottlenecks', [])
        for bottleneck in memory_bottlenecks[:2]:  # 前2个内存瓶颈
            optimizations.append({
                'type': 'memory',
                'priority': 'high' if bottleneck.get('size_mb', 0) > 100 else 'medium',
                'description': f"优化内存消耗: {bottleneck.get('consumer', '')}",
                'estimated_impact': f"减少 {bottleneck.get('size_mb', 0):.1f} MB 内存使用",
                'suggestions': [
                    "使用更紧凑的数据类型",
                    "分批处理数据",
                    "及时释放不再使用的对象"
                ]
            })

        return optimizations

    def _save_comparison_results(self, test_name: str) -> None:
        """保存比较结果"""
        if self.comparison_data.empty:
            return

        # 保存为CSV
        csv_path = self.output_dir / f"{test_name}_comparison.csv"
        self.comparison_data.to_csv(csv_path, index=False)

        # 保存为JSON
        json_path = self.output_dir / f"{test_name}_comparison.json"
        comparison_dict = self.comparison_data.to_dict('records')

        with open(json_path, 'w') as f:
            json.dump(comparison_dict, f, indent=2, default=str)

        print(f"比较结果已保存到: {csv_path}, {json_path}")

    def plot_comparison(self, test_name: str = "comparison") -> None:
        """
        绘制比较图表

        Args:
            test_name: 测试名称
        """
        if self.comparison_data.empty:
            print("没有比较数据可绘制")
            return

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 执行时间比较
        ax1 = axes[0, 0]
        implementations = self.comparison_data['implementation']
        times = self.comparison_data['execution_time']

        bars1 = ax1.bar(implementations, times, color='skyblue')
        ax1.set_title('执行时间比较')
        ax1.set_ylabel('时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + max(times) * 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. 内存使用比较
        ax2 = axes[0, 1]
        memory = self.comparison_data['memory_usage_mb']

        bars2 = ax2.bar(implementations, memory, color='lightgreen')
        ax2.set_title('内存使用比较')
        ax2.set_ylabel('内存 (MB)')
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(memory) * 0.01,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # 3. 速度提升
        ax3 = axes[1, 0]
        if 'time_speedup' in self.comparison_data.columns:
            speedup = self.comparison_data['time_speedup']

            bars3 = ax3.bar(implementations, speedup, color='lightcoral')
            ax3.set_title('速度提升 (相对于最快)')
            ax3.set_ylabel('速度提升倍数')
            ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax3.tick_params(axis='x', rotation=45)

        # 4. 综合评分
        ax4 = axes[1, 1]
        if 'score' in self.comparison_data.columns:
            scores = self.comparison_data['score']

            bars4 = ax4.bar(implementations, scores, color='gold')
            ax4.set_title('综合评分')
            ax4.set_ylabel('分数')
            ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # 保存图表
        plot_path = self.output_dir / f"{test_name}_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"比较图表已保存到: {plot_path}")


def compare_implementations(implementations: Dict[str, Callable],
                            test_data: np.ndarray,
                            test_name: str = "benchmark",
                            output_dir: str = "./results",
                            **kwargs) -> pd.DataFrame:
    """
    快速比较多个实现的便捷函数

    Args:
        implementations: 实现字典
        test_data: 测试数据
        test_name: 测试名称
        output_dir: 输出目录
        **kwargs: 参数

    Returns:
        比较结果DataFrame
    """
    analyzer = PerformanceAnalyzer(output_dir)
    return analyzer.compare_implementations(
        implementations, test_data, test_name, **kwargs
    )


def generate_performance_report(implementation_func: Callable,
                                test_data: np.ndarray,
                                implementation_name: str = "implementation",
                                output_dir: str = "./results",
                                **kwargs) -> Dict[str, Any]:
    """
    生成完整性能报告的便捷函数

    Args:
        implementation_func: 实现函数
        test_data: 测试数据
        implementation_name: 实现名称
        output_dir: 输出目录
        **kwargs: 参数

    Returns:
        性能报告
    """
    analyzer = PerformanceAnalyzer(output_dir)

    # 运行基准测试
    result = analyzer.benchmark_implementation(
        implementation_name, implementation_func, test_data, **kwargs
    )

    # 生成优化报告
    report = analyzer.generate_optimization_report(implementation_name)

    # 保存报告
    report_path = Path(output_dir) / f"{implementation_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"性能报告已保存到: {report_path}")

    return report