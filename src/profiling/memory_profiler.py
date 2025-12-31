"""
内存分析器
监控和优化DBSCAN算法的内存使用
"""

import tracemalloc
import psutil
import os
import gc
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import warnings
import sys


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    memory_usage_mb: float
    peak_memory_mb: float
    n_objects: int
    top_consumers: List[Tuple[str, float]]
    gc_stats: Dict[str, int]


class MemoryProfiler:
    """内存分析器"""

    def __init__(self, track_detailed: bool = False):
        """
        初始化内存分析器

        Args:
            track_detailed: 是否跟踪详细的内存分配信息
        """
        self.track_detailed = track_detailed
        self.snapshots: List[MemorySnapshot] = []
        self.start_time: Optional[float] = None
        self.process = psutil.Process(os.getpid())

        # 内存使用历史
        self.memory_history: List[Tuple[float, float]] = []
        self.peak_memory = 0.0

        # 对象计数
        self.object_counts = defaultdict(int)

    def start(self) -> None:
        """开始内存分析"""
        self.start_time = time.time()
        self.snapshots.clear()
        self.memory_history.clear()
        self.peak_memory = 0.0
        self.object_counts.clear()

        if self.track_detailed:
            tracemalloc.start(25)  # 跟踪25个帧

        print("内存分析已开始")

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """
        拍摄内存快照

        Args:
            label: 快照标签

        Returns:
            内存快照对象
        """
        current_time = time.time() - self.start_time if self.start_time else 0

        # 获取进程内存使用
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024  # 转换为MB

        # 更新峰值内存
        if memory_usage_mb > self.peak_memory:
            self.peak_memory = memory_usage_mb

        # 记录历史
        self.memory_history.append((current_time, memory_usage_mb))

        # 获取详细的内存分配信息
        top_consumers = []
        if self.track_detailed and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()

            # 获取内存消耗最大的代码位置
            top_stats = snapshot.statistics('lineno')[:10]

            for stat in top_stats:
                top_consumers.append((
                    stat.traceback.format()[-1] if stat.traceback else "Unknown",
                    stat.size / 1024 / 1024  # MB
                ))

        # 获取GC统计
        gc_stats = {
            'collected': gc.get_count()[0],
            'uncollectable': gc.get_count()[1],
            'threshold': gc.get_threshold()
        }

        # 获取对象计数（简化版）
        self._update_object_counts()

        # 创建快照
        snapshot = MemorySnapshot(
            timestamp=current_time,
            memory_usage_mb=memory_usage_mb,
            peak_memory_mb=self.peak_memory,
            n_objects=sum(self.object_counts.values()),
            top_consumers=top_consumers,
            gc_stats=gc_stats
        )

        self.snapshots.append(snapshot)

        if label:
            print(f"[{label}] 内存使用: {memory_usage_mb:.2f} MB, 峰值: {self.peak_memory:.2f} MB")

        return snapshot

    def _update_object_counts(self) -> None:
        """更新对象计数（简化实现）"""
        # 这里可以扩展为更详细的对象类型计数
        self.object_counts['total'] = len(gc.get_objects())

    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        分析函数的内存使用

        Args:
            func: 要分析的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            (函数结果, 内存分析结果)
        """
        self.start()

        # 开始前的快照
        before_snapshot = self.take_snapshot("开始前")

        # 执行函数
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # 执行后的快照
        after_snapshot = self.take_snapshot("结束后")

        # 停止分析
        self.stop()

        # 计算内存变化
        memory_increase = after_snapshot.memory_usage_mb - before_snapshot.memory_usage_mb

        # 分析结果
        analysis = {
            'function_name': func.__name__,
            'execution_time': execution_time,
            'memory_usage_before_mb': before_snapshot.memory_usage_mb,
            'memory_usage_after_mb': after_snapshot.memory_usage_mb,
            'memory_increase_mb': memory_increase,
            'peak_memory_mb': after_snapshot.peak_memory_mb,
            'n_snapshots': len(self.snapshots),
            'memory_leak_suspected': memory_increase > 10.0  # 超过10MB怀疑内存泄漏
        }

        return result, analysis

    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """
        分析内存使用模式

        Returns:
            内存模式分析结果
        """
        if len(self.snapshots) < 2:
            return {}

        # 转换为DataFrame便于分析
        data = []
        for snapshot in self.snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'memory_usage_mb': snapshot.memory_usage_mb,
                'n_objects': snapshot.n_objects
            })

        df = pd.DataFrame(data)

        # 分析内存增长模式
        memory_growth = df['memory_usage_mb'].diff().fillna(0)

        # 识别内存使用峰值
        peak_indices = np.where(memory_growth > memory_growth.mean() + 2 * memory_growth.std())[0]

        # 计算统计信息
        analysis = {
            'total_time': df['timestamp'].max() - df['timestamp'].min(),
            'avg_memory_usage_mb': df['memory_usage_mb'].mean(),
            'max_memory_usage_mb': df['memory_usage_mb'].max(),
            'min_memory_usage_mb': df['memory_usage_mb'].min(),
            'total_memory_growth_mb': df['memory_usage_mb'].iloc[-1] - df['memory_usage_mb'].iloc[0],
            'memory_growth_rate_mb_per_sec': (df['memory_usage_mb'].iloc[-1] - df['memory_usage_mb'].iloc[0]) /
                                             (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])
            if len(df) > 1 else 0,
            'n_memory_peaks': len(peak_indices),
            'peak_timestamps': df['timestamp'].iloc[peak_indices].tolist() if len(peak_indices) > 0 else [],
            'memory_usage_over_time': df[['timestamp', 'memory_usage_mb']].to_dict('records')
        }

        return analysis

    def find_memory_bottlenecks(self, threshold_mb: float = 50.0) -> List[Dict[str, Any]]:
        """
        查找内存瓶颈

        Args:
            threshold_mb: 内存消耗阈值(MB)

        Returns:
            瓶颈点列表
        """
        bottlenecks = []

        if not self.track_detailed or len(self.snapshots) < 2:
            return bottlenecks

        # 分析每个快照中的主要内存消费者
        for i, snapshot in enumerate(self.snapshots):
            for consumer, size_mb in snapshot.top_consumers:
                if size_mb > threshold_mb:
                    bottlenecks.append({
                        'snapshot_index': i,
                        'timestamp': snapshot.timestamp,
                        'consumer': consumer,
                        'size_mb': size_mb,
                        'total_memory_mb': snapshot.memory_usage_mb
                    })

        return bottlenecks

    def suggest_memory_optimizations(self) -> List[str]:
        """
        提供内存优化建议

        Returns:
            优化建议列表
        """
        suggestions = []

        # 分析内存使用模式
        patterns = self.analyze_memory_patterns()

        if not patterns:
            return suggestions

        # 检查内存增长
        if patterns['total_memory_growth_mb'] > 100:
            suggestions.append(
                f"检测到显著内存增长 ({patterns['total_memory_growth_mb']:.2f} MB)，可能存在内存泄漏"
            )

        # 检查内存峰值
        if patterns['n_memory_peaks'] > 5:
            suggestions.append(
                f"检测到多次内存峰值 ({patterns['n_memory_peaks']}次)，考虑使用内存池或分批处理"
            )

        # 检查大对象
        bottlenecks = self.find_memory_bottlenecks(threshold_mb=100)
        if bottlenecks:
            suggestions.append(
                f"发现 {len(bottlenecks)} 个内存瓶颈点，建议优化相关数据结构"
            )
            for bottleneck in bottlenecks[:3]:  # 只显示前3个
                suggestions.append(
                    f"  瓶颈点: {bottleneck['consumer']} 占用 {bottleneck['size_mb']:.2f} MB"
                )

        # 通用优化建议
        suggestions.extend([
            "考虑使用numpy数组代替Python列表存储数值数据",
            "使用适当的数据类型（如float32代替float64）",
            "及时释放不再使用的大型对象",
            "考虑使用内存映射文件处理超大数据集",
            "使用生成器代替列表以节省内存"
        ])

        return suggestions

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成内存分析报告

        Args:
            output_path: 报告输出路径（可选）

        Returns:
            报告数据字典
        """
        # 汇总分析结果
        patterns = self.analyze_memory_patterns()
        bottlenecks = self.find_memory_bottlenecks()
        suggestions = self.suggest_memory_optimizations()

        report = {
            'summary': {
                'n_snapshots': len(self.snapshots),
                'peak_memory_mb': self.peak_memory if self.snapshots else 0,
                'tracking_duration': patterns.get('total_time', 0) if patterns else 0
            },
            'memory_patterns': patterns,
            'bottlenecks': bottlenecks[:10],  # 只显示前10个
            'optimization_suggestions': suggestions,
            'snapshots_summary': [
                {
                    'timestamp': s.timestamp,
                    'memory_mb': s.memory_usage_mb,
                    'n_objects': s.n_objects
                }
                for s in self.snapshots[:20]  # 只显示前20个快照
            ]
        }

        # 保存报告到文件
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"内存分析报告已保存到: {output_path}")

        return report

    def stop(self) -> None:
        """停止内存分析"""
        if self.track_detailed and tracemalloc.is_tracing():
            tracemalloc.stop()

        print(f"内存分析已停止，峰值内存: {self.peak_memory:.2f} MB")

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


def track_memory_usage(func: Callable = None, interval: float = 0.1):
    """
    装饰器：跟踪函数内存使用

    Args:
        func: 要装饰的函数
        interval: 内存采样间隔（秒）
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            profiler = MemoryProfiler(track_detailed=True)

            with profiler:
                # 定期采样内存
                import threading
                stop_event = threading.Event()

                def monitor():
                    while not stop_event.is_set():
                        profiler.take_snapshot()
                        time.sleep(interval)

                # 启动监控线程
                monitor_thread = threading.Thread(target=monitor)
                monitor_thread.start()

                try:
                    result = f(*args, **kwargs)
                finally:
                    stop_event.set()
                    monitor_thread.join()

                # 生成分析报告
                report = profiler.generate_report()

                # 将报告附加到结果中
                if isinstance(result, tuple):
                    return result + (report,)
                else:
                    return result, report

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)

    return decorator if func is None else decorator(func)


def analyze_memory_patterns(data: np.ndarray, operations: List[str]) -> Dict[str, Any]:
    """
    分析特定操作的内存模式

    Args:
        data: 输入数据
        operations: 操作列表

    Returns:
        内存模式分析
    """
    profiler = MemoryProfiler(track_detailed=True)

    results = {}

    with profiler:
        for op in operations:
            if op == 'distance_matrix':
                profiler.take_snapshot("开始距离矩阵计算")
                # 模拟距离矩阵计算
                n = min(1000, len(data))
                sample = data[:n]
                dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        dist_matrix[i, j] = np.linalg.norm(sample[i] - sample[j])
                profiler.take_snapshot("结束距离矩阵计算")

                results[op] = {
                    'n_points': n,
                    'matrix_size_mb': dist_matrix.nbytes / 1024 / 1024
                }

            elif op == 'neighborhood_search':
                profiler.take_snapshot("开始邻域搜索")
                # 模拟邻域搜索
                n = min(500, len(data))
                sample = data[:n]
                eps = 0.01
                neighbors = []
                for i in range(n):
                    point_neighbors = []
                    for j in range(n):
                        if np.linalg.norm(sample[i] - sample[j]) <= eps:
                            point_neighbors.append(j)
                    neighbors.append(point_neighbors)
                profiler.take_snapshot("结束邻域搜索")

                results[op] = {
                    'n_points': n,
                    'avg_neighbors': np.mean([len(nbrs) for nbrs in neighbors])
                }

    # 生成总体分析
    patterns = profiler.analyze_memory_patterns()
    bottlenecks = profiler.find_memory_bottlenecks()

    return {
        'operations': results,
        'memory_patterns': patterns,
        'bottlenecks': bottlenecks,
        'suggestions': profiler.suggest_memory_optimizations()
    }