"""
性能分析脚本
分析串行和并行DBSCAN算法的性能差异
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import time
import argparse
import json
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

from src.data_processing.loader import TDriveDataLoader
from src.data_processing.preprocessor import TrajectoryPreprocessor
from src.clustering.dbscan_sequential import DBSCANSequential
from src.clustering.dbscan_parallel import DBSCANParallel
from src.profiling.time_profiler import TimeProfiler, profile_function
from src.profiling.memory_profiler import MemoryProfiler, track_memory_usage
from src.profiling.performance_analyzer import PerformanceAnalyzer, compare_implementations
from src.visualization.plot_performance import PerformanceVisualizer


def generate_test_data(n_points: int,
                       data_range: Tuple[float, float] = (39.0, 41.0)) -> np.ndarray:
    """
    生成测试数据

    Args:
        n_points: 点数
        data_range: 数据范围

    Returns:
        测试数据数组
    """
    print(f"生成 {n_points} 个测试点...")

    # 生成随机点
    np.random.seed(42)
    points = np.random.uniform(
        low=data_range[0],
        high=data_range[1],
        size=(n_points, 2)
    )

    return points


def load_real_data(data_path: str,
                   sample_rate: float = 0.001,
                   max_points: Optional[int] = None) -> np.ndarray:
    """
    加载真实轨迹数据

    Args:
        data_path: 数据路径
        sample_rate: 采样率
        max_points: 最大点数

    Returns:
        点数据数组
    """
    print(f"加载真实数据 (采样率: {sample_rate:.3%})...")

    # 创建数据加载器
    data_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
    loader = TDriveDataLoader(data_dir, max_workers=2)

    # 加载轨迹
    trajectories = loader.load_all_trajectories(
        sample_rate=sample_rate,
        limit_per_file=100
    )

    if not trajectories:
        raise ValueError("没有加载到轨迹数据")

    # 预处理
    preprocessor = TrajectoryPreprocessor(
        min_trajectory_length=5,
        max_speed_kmh=150.0,
        sampling_interval_s=120.0,
        remove_outliers=True
    )

    processed_trajectories = preprocessor.preprocess_trajectories(trajectories)

    # 提取点
    all_points = []
    for traj in processed_trajectories:
        for point in traj.points:
            all_points.append([point.latitude, point.longitude])

    points_array = np.array(all_points)

    if max_points and len(points_array) > max_points:
        indices = np.random.choice(len(points_array), max_points, replace=False)
        points_array = points_array[indices]

    print(f"加载了 {len(points_array)} 个点")

    return points_array


def profile_sequential_dbscan(points: np.ndarray,
                              eps: float = 0.01,
                              min_samples: int = 5,
                              metric: str = 'euclidean') -> Dict[str, any]:
    """
    分析串行DBSCAN性能

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数
        metric: 距离度量

    Returns:
        性能分析结果
    """
    print("\n" + "=" * 60)
    print("分析串行DBSCAN性能")
    print("=" * 60)

    # 创建DBSCAN聚类器
    dbscan = DBSCANSequential(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )

    # 使用装饰器分析性能
    @profile_function(detailed=True)
    def run_dbscan():
        return dbscan.fit(points)

    # 运行并获取性能分析
    result, time_analysis = run_dbscan()

    # 内存分析
    memory_profiler = MemoryProfiler(track_detailed=True)

    with memory_profiler:
        memory_before = memory_profiler.take_snapshot("开始前")
        dbscan.fit(points)
        memory_after = memory_profiler.take_snapshot("结束后")

    memory_usage = memory_after.memory_usage_mb - memory_before.memory_usage_mb
    memory_patterns = memory_profiler.analyze_memory_patterns()

    # 获取聚类统计
    stats = dbscan.get_cluster_stats()

    # 收集结果
    analysis_result = {
        'algorithm': 'DBSCAN_Sequential',
        'parameters': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'n_points': len(points)
        },
        'results': {
            'n_clusters': stats['n_clusters'],
            'n_core_points': stats['n_core_points'],
            'n_noise': stats['n_noise']
        },
        'performance': {
            'execution_time': time_analysis['execution_time'],
            'avg_time': time_analysis['avg_time'],
            'std_time': time_analysis['std_time'],
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': memory_after.peak_memory_mb,
            'memory_patterns': memory_patterns,
            'time_bottlenecks': time_analysis.get('top_functions', []),
            'memory_bottlenecks': memory_profiler.find_memory_bottlenecks()
        }
    }

    print(f"执行时间: {time_analysis['execution_time']:.4f} 秒")
    print(f"内存使用: {memory_usage:.2f} MB")
    print(f"峰值内存: {memory_after.peak_memory_mb:.2f} MB")

    return analysis_result


def profile_parallel_dbscan(points: np.ndarray,
                            eps: float = 0.01,
                            min_samples: int = 5,
                            metric: str = 'euclidean',
                            n_jobs: int = -1) -> Dict[str, any]:
    """
    分析并行DBSCAN性能

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数
        metric: 距离度量
        n_jobs: 工作进程数

    Returns:
        性能分析结果
    """
    print("\n" + "=" * 60)
    print(f"分析并行DBSCAN性能 (n_jobs={n_jobs})")
    print("=" * 60)

    # 设置工作进程数
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # 创建并行DBSCAN聚类器
    dbscan = DBSCANParallel(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
        chunk_size=1000
    )

    # 使用装饰器分析性能
    @profile_function(detailed=True)
    def run_parallel_dbscan():
        return dbscan.fit(points)

    # 运行并获取性能分析
    result, time_analysis = run_parallel_dbscan()

    # 内存分析
    memory_profiler = MemoryProfiler(track_detailed=True)

    with memory_profiler:
        memory_before = memory_profiler.take_snapshot("开始前")
        dbscan.fit(points)
        memory_after = memory_profiler.take_snapshot("结束后")

    memory_usage = memory_after.memory_usage_mb - memory_before.memory_usage_mb
    memory_patterns = memory_profiler.analyze_memory_patterns()

    # 获取聚类统计
    stats = dbscan.get_cluster_stats()
    perf_stats = dbscan.get_performance_stats()

    # 收集结果
    analysis_result = {
        'algorithm': f'DBSCAN_Parallel_{n_jobs}jobs',
        'parameters': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'n_points': len(points),
            'n_jobs': n_jobs
        },
        'results': {
            'n_clusters': stats['n_clusters'],
            'n_core_points': stats['n_core_points'],
            'n_noise': stats['n_noise']
        },
        'performance': {
            'execution_time': time_analysis['execution_time'],
            'avg_time': time_analysis['avg_time'],
            'std_time': time_analysis['std_time'],
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': memory_after.peak_memory_mb,
            'n_jobs': n_jobs,
            'memory_patterns': memory_patterns,
            'time_bottlenecks': time_analysis.get('top_functions', []),
            'memory_bottlenecks': memory_profiler.find_memory_bottlenecks()
        }
    }

    print(f"执行时间: {time_analysis['execution_time']:.4f} 秒")
    print(f"内存使用: {memory_usage:.2f} MB")
    print(f"峰值内存: {memory_after.peak_memory_mb:.2f} MB")
    print(f"工作进程: {n_jobs}")

    return analysis_result


def run_scalability_analysis(points: np.ndarray,
                             eps: float = 0.01,
                             min_samples: int = 5,
                             metric: str = 'euclidean',
                             max_workers: int = 8) -> Dict[str, any]:
    """
    运行可扩展性分析

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数
        metric: 距离度量
        max_workers: 最大工作进程数

    Returns:
        可扩展性分析结果
    """
    print("\n" + "=" * 60)
    print("运行可扩展性分析")
    print("=" * 60)

    scalability_results = {
        'n_workers': [],
        'execution_times': [],
        'speedups': [],
        'efficiencies': [],
        'memory_usages': []
    }

    # 运行串行版本作为基准
    print("运行串行版本作为基准...")
    serial_start = time.time()

    serial_dbscan = DBSCANSequential(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )
    serial_dbscan.fit(points)

    serial_time = time.time() - serial_start
    print(f"串行执行时间: {serial_time:.4f} 秒")

    # 测试不同工作进程数
    n_workers_list = list(range(1, min(max_workers, mp.cpu_count()) + 1))

    for n_workers in n_workers_list:
        print(f"\n测试 {n_workers} 个工作进程...")

        try:
            parallel_dbscan = DBSCANParallel(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                n_jobs=n_workers,
                chunk_size=1000
            )

            start_time = time.time()
            parallel_dbscan.fit(points)
            parallel_time = time.time() - start_time

            speedup = serial_time / parallel_time
            efficiency = speedup / n_workers

            scalability_results['n_workers'].append(n_workers)
            scalability_results['execution_times'].append(parallel_time)
            scalability_results['speedups'].append(speedup)
            scalability_results['efficiencies'].append(efficiency)

            print(f"  执行时间: {parallel_time:.4f} 秒")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  并行效率: {efficiency:.2%}")

        except Exception as e:
            print(f"  {n_workers} 个工作进程测试失败: {e}")
            continue

    return {
        'serial_time': serial_time,
        'scalability': scalability_results
    }


def run_comprehensive_comparison(points: np.ndarray,
                                 eps: float = 0.01,
                                 min_samples: int = 5,
                                 metric: str = 'euclidean') -> Dict[str, any]:
    """
    运行全面的性能比较

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数
        metric: 距离度量

    Returns:
        比较结果
    """
    print("\n" + "=" * 60)
    print("运行全面的性能比较")
    print("=" * 60)

    # 创建性能分析器
    analyzer = PerformanceAnalyzer(output_dir="./results/profiling")

    # 定义要比较的实现
    implementations = {
        'DBSCAN_Sequential': lambda data: DBSCANSequential(
            eps=eps, min_samples=min_samples, metric=metric
        ).fit(data),

        'DBSCAN_Parallel_2jobs': lambda data: DBSCANParallel(
            eps=eps, min_samples=min_samples, metric=metric, n_jobs=2
        ).fit(data),

        'DBSCAN_Parallel_4jobs': lambda data: DBSCANParallel(
            eps=eps, min_samples=min_samples, metric=metric, n_jobs=4
        ).fit(data),

        'DBSCAN_Parallel_8jobs': lambda data: DBSCANParallel(
            eps=eps, min_samples=min_samples, metric=metric, n_jobs=8
        ).fit(data),
    }

    # 运行比较
    comparison_results = compare_implementations(
        implementations=implementations,
        test_data=points,
        test_name="dbscan_comparison",
        output_dir="./results/profiling"
    )

    # 生成优化报告
    reports = {}
    for impl_name in implementations.keys():
        report = analyzer.generate_optimization_report(impl_name)
        if report:
            reports[impl_name] = report

    return {
        'comparison_results': comparison_results.to_dict('records'),
        'optimization_reports': reports
    }


def visualize_performance_analysis(all_results: Dict[str, any],
                                   scalability_results: Optional[Dict[str, any]] = None,
                                   comparison_results: Optional[Dict[str, any]] = None,
                                   output_dir: str = "./results") -> None:
    """
    可视化性能分析结果

    Args:
        all_results: 所有性能分析结果
        scalability_results: 可扩展性结果
        comparison_results: 比较结果
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("可视化性能分析结果")
    print("=" * 60)

    output_path = Path(output_dir) / "profiling"
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建可视化器
    visualizer = PerformanceVisualizer()

    # 1. 性能比较图
    if all_results:
        print("生成性能比较图...")

        # 准备比较数据
        comparison_data = {}
        for name, result in all_results.items():
            comparison_data[name] = {
                'execution_time': result['performance']['execution_time'],
                'memory_usage_mb': result['performance']['memory_usage_mb'],
                'peak_memory_mb': result['performance']['peak_memory_mb'],
                'accuracy': 1.0  # 假设准确性为1
            }

        visualizer.plot_execution_time_comparison(
            comparison_data,
            title="DBSCAN算法性能比较",
            save_path=str(output_path / "performance_comparison.png")
        )

    # 2. 可扩展性分析图
    if scalability_results:
        print("生成可扩展性分析图...")

        visualizer.plot_scalability_analysis(
            scalability_results['scalability'],
            title="DBSCAN并行可扩展性分析",
            save_path=str(output_path / "scalability_analysis.png")
        )

    # 3. 内存使用分析图
    if all_results and 'DBSCAN_Sequential' in all_results:
        print("生成内存使用分析图...")

        memory_data = all_results['DBSCAN_Sequential']['performance'].get('memory_patterns', {})
        if memory_data:
            visualizer.plot_memory_usage_analysis(
                memory_data,
                title="DBSCAN内存使用分析",
                save_path=str(output_path / "memory_analysis.png")
            )

    # 4. 并行效率分析图
    if scalability_results:
        print("生成并行效率分析图...")

        perf_data = {
            'parallel': {
                'n_workers': scalability_results['scalability']['n_workers'],
                'speedups': scalability_results['scalability']['speedups']
            }
        }

        visualizer.plot_parallel_efficiency(
            perf_data['parallel'],
            title="DBSCAN并行效率分析",
            save_path=str(output_path / "parallel_efficiency.png")
        )

    print(f"可视化结果已保存到: {output_path}")


def save_analysis_results(all_results: Dict[str, any],
                          scalability_results: Optional[Dict[str, any]] = None,
                          comparison_results: Optional[Dict[str, any]] = None,
                          output_dir: str = "./results") -> None:
    """
    保存性能分析结果

    Args:
        all_results: 所有性能分析结果
        scalability_results: 可扩展性结果
        comparison_results: 比较结果
        output_dir: 输出目录
    """
    output_path = Path(output_dir) / "profiling"
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存所有结果到JSON
    results_file = output_path / "performance_analysis.json"

    all_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'system_info': {
            'cpu_count': mp.cpu_count(),
            'platform': sys.platform,
            'python_version': sys.version
        },
        'individual_results': all_results,
        'scalability_results': scalability_results,
        'comparison_results': comparison_results
    }

    with open(results_file, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

    # 保存摘要
    summary_file = output_path / "performance_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DBSCAN算法性能分析摘要\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"分析时间: {all_data['timestamp']}\n")
        f.write(f"系统信息: {mp.cpu_count()} CPU核心, {sys.platform}\n\n")

        if all_results:
            f.write("各算法性能:\n")
            f.write("-" * 40 + "\n")

            for name, result in all_results.items():
                perf = result['performance']
                f.write(f"\n{name}:\n")
                f.write(f"  执行时间: {perf['execution_time']:.4f} 秒\n")
                f.write(f"  内存使用: {perf['memory_usage_mb']:.2f} MB\n")
                f.write(f"  峰值内存: {perf['peak_memory_mb']:.2f} MB\n")
                if 'n_jobs' in perf:
                    f.write(f"  工作进程: {perf['n_jobs']}\n")

        if scalability_results:
            f.write("\n\n可扩展性分析:\n")
            f.write("-" * 40 + "\n")
            f.write(f"串行执行时间: {scalability_results['serial_time']:.4f} 秒\n")

            for i, n_workers in enumerate(scalability_results['scalability']['n_workers']):
                speedup = scalability_results['scalability']['speedups'][i]
                efficiency = scalability_results['scalability']['efficiencies'][i]
                f.write(f"{n_workers} 个工作进程: 加速比={speedup:.2f}x, 效率={efficiency:.2%}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("性能优化建议:\n")
        f.write("-" * 40 + "\n")

        # 添加优化建议
        if all_results and 'DBSCAN_Sequential' in all_results:
            perf = all_results['DBSCAN_Sequential']['performance']
            if 'time_bottlenecks' in perf and perf['time_bottlenecks']:
                f.write("\n时间瓶颈:\n")
                for bottleneck in perf['time_bottlenecks'][:3]:
                    f.write(f"  {bottleneck.get('function', 'Unknown')}\n")

            if 'memory_bottlenecks' in perf and perf['memory_bottlenecks']:
                f.write("\n内存瓶颈:\n")
                for bottleneck in perf['memory_bottlenecks'][:3]:
                    f.write(f"  {bottleneck.get('consumer', 'Unknown')}: {bottleneck.get('size_mb', 0):.1f} MB\n")

    print(f"分析结果已保存到: {results_file}")
    print(f"摘要已保存到: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DBSCAN算法性能分析')
    parser.add_argument('--data', type=str,
                        help='T-Drive数据路径（文件或目录），如果未指定则使用生成的数据')
    parser.add_argument('--synthetic', action='store_true',
                        help='使用生成的数据进行测试')
    parser.add_argument('--n-points', type=int, default=10000,
                        help='生成数据时的点数（默认: 10000）')
    parser.add_argument('--sample-rate', type=float, default=0.001,
                        help='真实数据采样率（默认: 0.001）')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='DBSCAN邻域半径（默认: 0.01）')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN最小样本数（默认: 5）')
    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['euclidean', 'haversine'],
                        help='距离度量方式（默认: euclidean）')
    parser.add_argument('--n-jobs', type=int, nargs='+', default=[-1],
                        help='并行测试的工作进程数，可以指定多个（默认: [-1]使用所有核心）')
    parser.add_argument('--run-scalability', action='store_true',
                        help='运行可扩展性分析')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='可扩展性分析的最大工作进程数（默认: 8）')
    parser.add_argument('--run-comparison', action='store_true',
                        help='运行全面的性能比较')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='输出目录（默认: ./results）')
    parser.add_argument('--no-visualize', action='store_true',
                        help='不生成可视化图表')

    args = parser.parse_args()

    try:
        print("DBSCAN算法性能分析")
        print("=" * 60)

        # 1. 加载或生成测试数据
        if args.synthetic:
            print("使用生成的数据进行测试...")
            points = generate_test_data(args.n_points)
        elif args.data:
            print("使用真实数据进行测试...")
            points = load_real_data(args.data, args.sample_rate, args.n_points)
        else:
            print("使用默认生成的数据进行测试...")
            points = generate_test_data(args.n_points)

        print(f"测试数据: {len(points)} 个点")

        # 2. 分析各算法性能
        all_results = {}

        # 串行版本
        print("\n分析串行版本...")
        serial_result = profile_sequential_dbscan(
            points, args.eps, args.min_samples, args.metric
        )
        all_results['DBSCAN_Sequential'] = serial_result

        # 并行版本
        for n_jobs in args.n_jobs:
            print(f"\n分析并行版本 (n_jobs={n_jobs})...")
            parallel_result = profile_parallel_dbscan(
                points, args.eps, args.min_samples, args.metric, n_jobs
            )
            all_results[f'DBSCAN_Parallel_{n_jobs}jobs'] = parallel_result

        # 3. 可扩展性分析（可选）
        scalability_results = None
        if args.run_scalability:
            scalability_results = run_scalability_analysis(
                points, args.eps, args.min_samples, args.metric, args.max_workers
            )

        # 4. 全面性能比较（可选）
        comparison_results = None
        if args.run_comparison:
            comparison_results = run_comprehensive_comparison(
                points, args.eps, args.min_samples, args.metric
            )

        # 5. 可视化结果
        if not args.no_visualize:
            visualize_performance_analysis(
                all_results, scalability_results, comparison_results, args.output_dir
            )

        # 6. 保存结果
        save_analysis_results(
            all_results, scalability_results, comparison_results, args.output_dir
        )

        print("\n" + "=" * 60)
        print("性能分析完成")
        print("=" * 60)

        # 打印关键发现
        print("\n关键发现:")

        if len(all_results) > 1:
            # 找出最快的实现
            fastest_name = min(all_results.items(),
                               key=lambda x: x[1]['performance']['execution_time'])[0]
            fastest_time = all_results[fastest_name]['performance']['execution_time']

            # 找出最慢的实现
            slowest_name = max(all_results.items(),
                               key=lambda x: x[1]['performance']['execution_time'])[0]
            slowest_time = all_results[slowest_name]['performance']['execution_time']

            speedup_ratio = slowest_time / fastest_time if fastest_time > 0 else 0

            print(f"  最快实现: {fastest_name} ({fastest_time:.4f} 秒)")
            print(f"  最慢实现: {slowest_name} ({slowest_time:.4f} 秒)")
            print(f"  加速比: {speedup_ratio:.2f}x")

        if scalability_results:
            best_idx = np.argmax(scalability_results['scalability']['speedups'])
            best_workers = scalability_results['scalability']['n_workers'][best_idx]
            best_speedup = scalability_results['scalability']['speedups'][best_idx]
            print(f"  最优并行配置: {best_workers} 个工作进程")
            print(f"  最大加速比: {best_speedup:.2f}x")

        print(f"\n详细结果已保存到: {Path(args.output_dir) / 'profiling'}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()