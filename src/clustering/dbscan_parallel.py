"""
并行DBSCAN实现
利用多核CPU加速聚类过程
"""

import numpy as np
from typing import List, Tuple, Dict
import multiprocessing as mp
from multiprocessing import Pool, shared_memory
import time
import warnings
from collections import defaultdict

try:
    from sklearn.neighbors import BallTree

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Using slower distance calculation.")


class DBSCANParallel:
    """并行版本的DBSCAN聚类算法"""

    def __init__(self, eps: float = 100.0, min_samples: int = 5,
                 metric: str = 'euclidean', n_jobs: int = -1,
                 chunk_size: int = 1000):
        """
        初始化并行DBSCAN参数

        Args:
            eps: 邻域半径（米）
            min_samples: 核心点的最小邻居数
            metric: 距离度量方式
            n_jobs: 并行工作进程数，-1表示使用所有CPU核心
            chunk_size: 每个工作进程处理的数据块大小
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
        self.chunk_size = chunk_size

        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.execution_time = 0
        self.parallel_time = 0

    def fit(self, points: np.ndarray) -> 'DBSCANParallel':
        """
        执行并行DBSCAN聚类

        Args:
            points: 形状为(n_samples, 2)的numpy数组

        Returns:
            self: 返回聚类器实例
        """
        start_time = time.time()
        n_samples = points.shape[0]

        # 创建共享内存
        shm, shared_points = self._create_shared_array(points)

        try:
            # 划分数据块
            chunks = self._partition_data(n_samples)

            # 并行查找邻域
            print(f"使用 {self.n_jobs} 个进程进行并行计算...")
            with Pool(processes=self.n_jobs) as pool:
                # 第一步：并行查找所有点的邻居
                neighbor_tasks = []
                for chunk_start, chunk_end in chunks:
                    task = pool.apply_async(
                        self._find_neighbors_chunk,
                        args=(chunk_start, chunk_end, n_samples, self.eps,
                              self.min_samples, self.metric)
                    )
                    neighbor_tasks.append(task)

                # 收集邻居信息
                all_neighbors = []
                core_points = []

                for i, task in enumerate(neighbor_tasks):
                    chunk_neighbors, chunk_core_points = task.get()
                    all_neighbors.extend(chunk_neighbors)
                    core_points.extend(chunk_core_points)

            # 第二步：并行扩展聚类
            labels = self._parallel_cluster_expansion(
                all_neighbors, core_points, n_samples
            )

        finally:
            # 清理共享内存
            shm.close()
            shm.unlink()

        # 保存结果
        self.labels_ = np.array(labels, dtype=np.int32)
        self.core_sample_indices_ = np.where(self.labels_ > 0)[0]
        self.components_ = points[self.core_sample_indices_]

        self.execution_time = time.time() - start_time

        return self

    def _create_shared_array(self, points: np.ndarray) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
        """
        创建共享内存数组

        Args:
            points: 原始数据点

        Returns:
            (共享内存对象, 共享数组视图)
        """
        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=points.nbytes)

        # 在共享内存中创建数组
        shared_array = np.ndarray(
            points.shape, dtype=points.dtype, buffer=shm.buf
        )
        shared_array[:] = points[:]  # 复制数据到共享内存

        return shm, shared_array

    def _partition_data(self, n_samples: int) -> List[Tuple[int, int]]:
        """
        将数据划分为多个块

        Args:
            n_samples: 总样本数

        Returns:
            数据块列表，每个元素是(start_idx, end_idx)元组
        """
        chunks = []
        for start_idx in range(0, n_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_samples)
            chunks.append((start_idx, end_idx))
        return chunks

    @staticmethod
    def _find_neighbors_chunk(chunk_start: int, chunk_end: int, n_samples: int,
                              eps: float, min_samples: int, metric: str) -> Tuple[List[List[int]], List[int]]:
        """
        处理一个数据块的邻居查找（工作进程函数）

        Args:
            chunk_start: 数据块起始索引
            chunk_end: 数据块结束索引
            n_samples: 总样本数
            eps: 邻域半径
            min_samples: 最小样本数
            metric: 距离度量方式

        Returns:
            (邻居列表, 核心点列表)
        """
        # 重新连接共享内存
        try:
            existing_shm = shared_memory.SharedMemory(name='shared_points')
            points = np.ndarray((n_samples, 2), dtype=np.float64, buffer=existing_shm.buf)
        except:
            # 回退到普通数组（用于测试）
            return [], []

        chunk_neighbors = []
        chunk_core_points = []

        for i in range(chunk_start, chunk_end):
            neighbors = []
            point = points[i]

            for j in range(n_samples):
                if i == j:
                    continue

                # 计算距离（简化版）
                if metric == 'euclidean':
                    distance = np.sqrt(np.sum((point - points[j]) ** 2))
                else:
                    # Haversine距离计算
                    from math import radians, sin, cos, sqrt, atan2
                    lat1, lon1 = radians(point[0]), radians(point[1])
                    lat2, lon2 = radians(points[j][0]), radians(points[j][1])
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    R = 6371000
                    distance = R * c

                if distance <= eps:
                    neighbors.append(j)

            chunk_neighbors.append(neighbors)
            if len(neighbors) >= min_samples:
                chunk_core_points.append(i)

        return chunk_neighbors, chunk_core_points

    def _parallel_cluster_expansion(self, all_neighbors: List[List[int]],
                                    core_points: List[int], n_samples: int) -> List[int]:
        """
        并行扩展聚类

        Args:
            all_neighbors: 所有点的邻居列表
            core_points: 核心点索引列表
            n_samples: 总样本数

        Returns:
            聚类标签列表
        """
        # 初始化标签
        labels = [-1] * n_samples
        cluster_id = 1

        # 为每个核心点创建聚类
        for core_idx in core_points:
            if labels[core_idx] != -1:
                continue

            # 创建新的聚类
            seeds = all_neighbors[core_idx].copy()
            labels[core_idx] = cluster_id

            # 扩展聚类
            i = 0
            while i < len(seeds):
                point_idx = seeds[i]

                if labels[point_idx] == 0:
                    labels[point_idx] = cluster_id

                if labels[point_idx] == -1:
                    labels[point_idx] = cluster_id

                    # 如果该点是核心点，添加其邻居
                    if point_idx in core_points:
                        for neighbor_idx in all_neighbors[point_idx]:
                            if labels[neighbor_idx] <= 0 and neighbor_idx not in seeds:
                                seeds.append(neighbor_idx)

                i += 1

            cluster_id += 1

        # 标记噪声点
        for i in range(n_samples):
            if labels[i] == -1:
                labels[i] = 0

        return labels

    def get_performance_stats(self) -> dict:
        """
        获取性能统计信息

        Returns:
            性能统计字典
        """
        stats = self.get_cluster_stats()
        stats.update({
            'n_jobs': self.n_jobs,
            'chunk_size': self.chunk_size,
            'parallel_time': self.parallel_time
        })
        return stats

    def get_cluster_stats(self) -> dict:
        """
        获取聚类统计信息（与串行版本兼容）
        """
        if self.labels_ is None:
            return {}

        unique_labels = np.unique(self.labels_)
        stats = {
            'n_clusters': len(unique_labels) - (1 if 0 in unique_labels else 0),
            'n_noise': np.sum(self.labels_ == 0),
            'n_core_points': len(self.core_sample_indices_) if self.core_sample_indices_ is not None else 0,
            'execution_time': self.execution_time,
            'cluster_sizes': {}
        }

        for label in unique_labels:
            if label != 0:
                stats['cluster_sizes'][label] = np.sum(self.labels_ == label)

        return stats