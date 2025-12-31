#!/usr/bin/env python3
"""
运行并行DBSCAN聚类算法
利用多核CPU进行加速
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
from typing import Dict, List, Optional

from src.data_processing.loader import TDriveDataLoader, load_tdrive_csv
from src.data_processing.preprocessor import TrajectoryPreprocessor
from src.clustering.dbscan_parallel import DBSCANParallel
from src.parallel.partitioning import GridPartitioner, KDTreePartitioner
from src.parallel.workers import WorkerPool
from src.profiling.time_profiler import TimeProfiler
from src.profiling.memory_profiler import MemoryProfiler
from src.profiling.performance_analyzer import PerformanceAnalyzer
from src.visualization.plot_clusters import ClusterVisualizer
from src.visualization.plot_performance import PerformanceVisualizer


def load_and_preprocess_data(data_path: str,
                             sample_rate: float = 0.01,
                             n_trajectories: Optional[int] = None,
                             max_points: Optional[int] = None) -> np.ndarray:
    """
    加载和预处理轨迹数据（并行版本）

    Args:
        data_path: 数据路径
        sample_rate: 数据采样率
        n_trajectories: 轨迹数量限制
        max_points: 最大点数限制

    Returns:
        预处理后的点数据
    """
    print("=" * 60)
    print("数据加载和预处理（并行版本）")
    print("=" * 60)

    start_time = time.time()

    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 创建数据加载器
    data_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
    loader = TDriveDataLoader(data_dir, max_workers=mp.cpu_count())

    # 获取数据集信息
    info = loader.get_dataset_info()
    print(f"数据集信息:")
    print(f"  文件数: {info['n_files']}")
    print(f"  总大小: {info['total_size_mb']:.2f} MB")
    print(f"  数据目录: {info['data_dir']}")
    print(f"  可用CPU核心: {mp.cpu_count()}")

    # 加载轨迹数据（并行）
    print(f"\n并行加载数据 (采样率: {sample_rate:.1%})...")
    trajectories = loader.load_all_trajectories(
        sample_rate=sample_rate,
        limit_per_file=n_trajectories
    )

    if not trajectories:
        raise ValueError("没有加载到任何轨迹数据")

    print(f"加载了 {len(trajectories)} 条轨迹")

    # 预处理轨迹
    print("\n预处理轨迹数据...")
    preprocessor = TrajectoryPreprocessor(
        min_trajectory_length=10,
        max_speed_kmh=120.0,
        sampling_interval_s=60.0,
        remove_outliers=True
    )

    processed_trajectories = preprocessor.preprocess_trajectories(trajectories)
    print(f"预处理后保留 {len(processed_trajectories)} 条轨迹")

    # 提取所有点
    all_points = []
    for traj in processed_trajectories:
        for point in traj.points:
            all_points.append([point.latitude, point.longitude])

    points_array = np.array(all_points)

    # 如果设置了最大点数，进行采样
    if max_points and len(points_array) > max_points:
        print(f"采样到 {max_points} 个点")
        indices = np.random.choice(len(points_array), max_points, replace=False)
        points_array = points_array[indices]

    print(f"提取了 {len(points_array)} 个轨迹点")

    loading_time = time.time() - start_time
    print(f"数据加载和预处理耗时: {loading_time:.2f} 秒")

    return points_array


def run_parallel_dbscan(points: np.ndarray,
                        eps: float = 0.01,
                        min_samples: int = 5,
                        metric: str = 'euclidean',
                        n_jobs: int = -1,
                        chunk_size: int = 1000,
                        partition_method: str = 'grid') -> Dict[str, any]:
    """
    运行并行DBSCAN算法

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数
        metric: 距离度量
        n_jobs: 并行工作进程数
        chunk_size: 数据块大小
        partition_method: 分区方法

    Returns:
        聚类结果和性能数据
    """
    print("\n" + "=" * 60)
    print("运行并行DBSCAN聚类")
    print("=" * 60)

    # 设置工作进程数
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # 创建性能分析器
    time_profiler = TimeProfiler(enable_profiling=True)
    memory_profiler = MemoryProfiler(track_detailed=True)

    # 开始内存分析
    memory_profiler.start()

    print(f"算法参数:")
    print(f"  eps (邻域半径): {eps}")
    print(f"  min_samples (最小样本数): {min_samples}")
    print(f"  metric (距离度量): {metric}")
    print(f"  数据点数量: {len(points)}")
    print(f"  工作进程数: {n_jobs}")
    print(f"  数据块大小: {chunk_size}")
    print(f"  分区方法: {partition_method}")

    # 创建并行DBSCAN聚类器
    dbscan = DBSCANParallel(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
        chunk_size=chunk_size
    )

    # 分析聚类性能
    with memory_profiler:
        memory_before = memory_profiler.take_snapshot("聚类开始前")

        # 执行聚类
        start_time = time.time()
        dbscan.fit(points)
        execution_time = time.time() - start_time

        memory_after = memory_profiler.take_snapshot("聚类结束后")
        memory_usage = memory_after.memory_usage_mb - memory_before.memory_usage_mb

    # 获取聚类统计信息
    stats = dbscan.get_cluster_stats()
    perf_stats = dbscan.get_performance_stats()

    print(f"\n聚类结果:")
    print(f"  聚类数量: {stats['n_clusters']}")
    print(f"  核心点数量: {stats['n_core_points']}")
    print(f"  噪声点数量: {stats['n_noise']}")
    print(f"  总点数: {len(points)}")

    if stats['cluster_sizes']:
        print(f"  聚类大小分布:")
        for label, size in list(stats['cluster_sizes'].items())[:10]:
            print(f"    聚类 {label}: {size} 个点")
        if len(stats['cluster_sizes']) > 10:
            print(f"    ... 还有 {len(stats['cluster_sizes']) - 10} 个聚类")

    print(f"\n性能统计:")
    print(f"  执行时间: {execution_time:.4f} 秒")
    print(f"  并行时间: {perf_stats.get('parallel_time', 0):.4f} 秒")
    print(f"  内存使用: {memory_usage:.2f} MB")
    print(f"  峰值内存: {memory_after.peak_memory_mb:.2f} MB")
    print(f"  工作进程: {perf_stats['n_jobs']}")

    # 分析内存模式
    memory_patterns = memory_profiler.analyze_memory_patterns()

    # 分析时间瓶颈
    time_analysis = time_profiler._analyze_function_performance('DBSCAN.fit')

    # 收集结果
    result = {
        'algorithm': 'DBSCAN_Parallel',
        'parameters': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'n_points': len(points),
            'n_jobs': n_jobs,
            'chunk_size': chunk_size,
            'partition_method': partition_method
        },
        'results': {
            'n_clusters': stats['n_clusters'],
            'n_core_points': stats['n_core_points'],
            'n_noise': stats['n_noise'],
            'cluster_sizes': stats['cluster_sizes'],
            'labels': dbscan.labels_.tolist() if dbscan.labels_ is not None else []
        },
        'performance': {
            'execution_time': execution_time,
            'parallel_time': perf_stats.get('parallel_time', 0),
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': memory_after.peak_memory_mb,
            'n_jobs': n_jobs,
            'memory_patterns': memory_patterns,
            'time_analysis': time_analysis
        },
        'dbscan_object': dbscan
    }

    return result


def run_scalability_test(points: np.ndarray,
                         eps: float = 0.01,
                         min_samples: int = 5,
                         metric: str = 'euclidean',
                         max_workers: int = 8) -> Dict[str, any]:
    """
    运行可扩展性测试（不同工作进程数的性能）

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数
        metric: 距离度量
        max_workers: 最大工作进程数

    Returns:
        可扩展性测试结果
    """
    print("\n" + "=" * 60)
    print("运行可扩展性测试")
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
    from src.clustering.dbscan_sequential import DBSCANSequential

    serial_dbscan = DBSCANSequential(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )

    serial_start = time.time()
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


def visualize_parallel_results(points: np.ndarray, result: Dict[str, any],
                               scalability_results: Optional[Dict[str, any]] = None,
                               output_dir: str = "./results") -> None:
    """
    可视化并行聚类结果

    Args:
        points: 点数据
        result: 聚类结果
        scalability_results: 可扩展性测试结果
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("可视化并行聚类结果")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建可视化器
    cluster_visualizer = ClusterVisualizer(figsize=(14, 12))
    perf_visualizer = PerformanceVisualizer()

    labels = np.array(result['results']['labels'])

    # 1. 2D聚类图
    print("生成2D聚类图...")
    fig1 = cluster_visualizer.plot_clusters_2d(
        points, labels,
        title=f"并行DBSCAN聚类结果 (eps={result['parameters']['eps']}, workers={result['parameters']['n_jobs']})",
        save_path=str(output_path / "parallel_clusters_2d.png")
    )

    # 2. 聚类热力图
    print("生成聚类热力图...")
    fig2 = cluster_visualizer.plot_cluster_heatmap(
        points, labels,
        save_path=str(output_path / "parallel_clusters_heatmap.png")
    )

    # 3. 并行性能分析
    if scalability_results:
        print("生成并行性能分析图...")

        # 创建性能数据
        perf_data = {
            'parallel': {
                'n_workers': scalability_results['scalability']['n_workers'],
                'speedups': scalability_results['scalability']['speedups'],
                'efficiencies': scalability_results['scalability']['efficiencies']
            }
        }

        fig3 = perf_visualizer.plot_parallel_efficiency(
            perf_data['parallel'],
            title=f"并行效率分析 (数据点: {len(points)})",
            save_path=str(output_path / "parallel_efficiency.png")
        )

    print(f"可视化结果已保存到: {output_path}")


def save_parallel_results(result: Dict[str, any],
                          scalability_results: Optional[Dict[str, any]] = None,
                          output_dir: str = "./results") -> None:
    """
    保存并行聚类结果

    Args:
        result: 聚类结果
        scalability_results: 可扩展性测试结果
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存完整结果到JSON
    result_file = output_path / "parallel_results.json"

    # 准备可序列化的结果
    serializable_result = {
        'algorithm': result['algorithm'],
        'parameters': result['parameters'],
        'results': {
            'n_clusters': result['results']['n_clusters'],
            'n_core_points': result['results']['n_core_points'],
            'n_noise': result['results']['n_noise'],
            'cluster_sizes': result['results']['cluster_sizes']
        },
        'performance': {
            'execution_time': result['performance']['execution_time'],
            'parallel_time': result['performance'].get('parallel_time', 0),
            'memory_usage_mb': result['performance']['memory_usage_mb'],
            'peak_memory_mb': result['performance']['peak_memory_mb'],
            'n_jobs': result['performance']['n_jobs']
        }
    }

    if scalability_results:
        serializable_result['scalability'] = scalability_results

    with open(result_file, 'w') as f:
        json.dump(serializable_result, f, indent=2, default=str)

    # 保存性能摘要
    summary_file = output_path / "parallel_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("并行DBSCAN聚类结果摘要\n")
        f.write("=" * 50 + "\n\n")

        f.write("算法参数:\n")
        for key, value in result['parameters'].items():
            f.write(f"  {key}: {value}\n")

        f.write("\n聚类结果:\n")
        f.write(f"  聚类数量: {result['results']['n_clusters']}\n")
        f.write(f"  核心点数量: {result['results']['n_core_points']}\n")
        f.write(f"  噪声点数量: {result['results']['n_noise']}\n")

        f.write("\n性能统计:\n")
        f.write(f"  执行时间: {result['performance']['execution_time']:.4f} 秒\n")
        f.write(f"  并行时间: {result['performance'].get('parallel_time', 0):.4f} 秒\n")
        f.write(f"  内存使用: {result['performance']['memory_usage_mb']:.2f} MB\n")
        f.write(f"  峰值内存: {result['performance']['peak_memory_mb']:.2f} MB\n")
        f.write(f"  工作进程数: {result['performance']['n_jobs']}\n")

        if scalability_results:
            f.write(f"\n可扩展性测试:\n")
            f.write(f"  串行执行时间: {scalability_results['serial_time']:.4f} 秒\n")

            for i, n_workers in enumerate(scalability_results['scalability']['n_workers']):
                speedup = scalability_results['scalability']['speedups'][i]
                efficiency = scalability_results['scalability']['efficiencies'][i]
                f.write(f"  {n_workers} 个工作进程: 加速比={speedup:.2f}x, 效率={efficiency:.2%}\n")

    print(f"结果已保存到: {result_file}")
    print(f"摘要已保存到: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行并行DBSCAN聚类算法')
    parser.add_argument('--data', type=str, required=True,
                        help='T-Drive数据路径（文件或目录）')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='DBSCAN邻域半径（默认: 0.01）')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN最小样本数（默认: 5）')
    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['euclidean', 'haversine'],
                        help='距离度量方式（默认: euclidean）')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='并行工作进程数，-1表示使用所有CPU核心（默认: -1）')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='数据块大小（默认: 1000）')
    parser.add_argument('--sample-rate', type=float, default=0.01,
                        help='数据采样率（0-1，默认: 0.01）')
    parser.add_argument('--max-points', type=int,
                        help='最大点数限制（用于测试）')
    parser.add_argument('--output-dir', type=str, default='./results/parallel',
                        help='输出目录（默认: ./results/parallel）')
    parser.add_argument('--run-scalability', action='store_true',
                        help='运行可扩展性测试')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='可扩展性测试的最大工作进程数（默认: 8）')
    parser.add_argument('--no-visualize', action='store_true',
                        help='不生成可视化图表')

    args = parser.parse_args()

    try:
        print("并行DBSCAN聚类算法")
        print("=" * 60)

        # 1. 加载和预处理数据
        points = load_and_preprocess_data(
            args.data,
            sample_rate=args.sample_rate,
            max_points=args.max_points
        )

        # 2. 运行并行DBSCAN
        result = run_parallel_dbscan(
            points,
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size
        )

        # 3. 运行可扩展性测试（可选）
        scalability_results = None
        if args.run_scalability:
            scalability_results = run_scalability_test(
                points,
                eps=args.eps,
                min_samples=args.min_samples,
                metric=args.metric,
                max_workers=args.max_workers
            )

        # 4. 可视化结果
        if not args.no_visualize:
            visualize_parallel_results(points, result, scalability_results, args.output_dir)

        # 5. 保存结果
        save_parallel_results(result, scalability_results, args.output_dir)

        print("\n" + "=" * 60)
        print("并行DBSCAN聚类完成")
        print("=" * 60)

        # 打印摘要
        print(f"\n聚类摘要:")
        print(f"  数据点: {len(points)}")
        print(f"  聚类数: {result['results']['n_clusters']}")
        print(f"  噪声点: {result['results']['n_noise']}")
        print(f"  执行时间: {result['performance']['execution_time']:.4f} 秒")
        print(f"  内存使用: {result['performance']['memory_usage_mb']:.2f} MB")
        print(f"  工作进程: {result['performance']['n_jobs']}")

        if scalability_results:
            print(f"\n可扩展性测试摘要:")
            print(f"  串行时间: {scalability_results['serial_time']:.4f} 秒")
            best_idx = np.argmax(scalability_results['scalability']['speedups'])
            best_workers = scalability_results['scalability']['n_workers'][best_idx]
            best_speedup = scalability_results['scalability']['speedups'][best_idx]
            print(f"  最佳配置: {best_workers} 个工作进程，加速比 {best_speedup:.2f}x")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()