"""
串行DBSCAN实现
经典的密度聚类算法
"""

import numpy as np
from typing import List, Tuple, Set
import time
from ..data_processing.trajectory import TrajectoryPoint


class DBSCANSequential:
    """串行版本的DBSCAN聚类算法"""

    def __init__(self, eps: float = 100.0, min_samples: int = 5,
                 metric: str = 'euclidean'):
        """
        初始化DBSCAN参数

        Args:
            eps: 邻域半径（米）
            min_samples: 核心点的最小邻居数
            metric: 距离度量方式，支持'euclidean'和'haversine'
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.execution_time = 0

    def fit(self, points: np.ndarray) -> 'DBSCANSequential':
        """
        执行DBSCAN聚类

        Args:
            points: 形状为(n_samples, 2)的numpy数组，每一行是[latitude, longitude]

        Returns:
            self: 返回聚类器实例
        """
        start_time = time.time()

        n_samples = points.shape[0]

        # 初始化标签：-1表示未访问，0表示噪声
        labels = np.full(n_samples, -1, dtype=np.int32)

        # 核心点索引集合
        core_indices = []
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1:  # 已访问的点
                continue

            # 查找邻域内的点
            neighbors = self._find_neighbors(points, i)

            if len(neighbors) < self.min_samples:
                # 标记为噪声点
                labels[i] = 0
                continue

            # 发现核心点，开始新的聚类
            labels[i] = cluster_id
            core_indices.append(i)

            # 扩展聚类
            self._expand_cluster(points, labels, neighbors, cluster_id)

            cluster_id += 1

        # 保存结果
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_indices, dtype=np.int32)
        self.components_ = points[self.core_sample_indices_]
        self.execution_time = time.time() - start_time

        return self

    def _find_neighbors(self, points: np.ndarray, point_idx: int) -> List[int]:
        """
        查找指定点邻域内的所有点

        Args:
            points: 所有点的数组
            point_idx: 目标点的索引

        Returns:
            邻域内点的索引列表
        """
        neighbors = []
        point = points[point_idx]

        for i in range(points.shape[0]):
            if i == point_idx:
                continue

            distance = self._compute_distance(point, points[i])
            if distance <= self.eps:
                neighbors.append(i)

        return neighbors

    def _expand_cluster(self, points: np.ndarray, labels: np.ndarray,
                        seeds: List[int], cluster_id: int):
        """
        从种子点扩展聚类

        Args:
            points: 所有点的数组
            labels: 标签数组
            seeds: 种子点索引列表
            cluster_id: 当前聚类ID
        """
        i = 0
        while i < len(seeds):
            point_idx = seeds[i]

            if labels[point_idx] == 0:  # 如果之前标记为噪声，重新标记
                labels[point_idx] = cluster_id

            if labels[point_idx] == -1:  # 未访问的点
                labels[point_idx] = cluster_id

                # 查找该点的邻居
                neighbors = self._find_neighbors(points, point_idx)

                if len(neighbors) >= self.min_samples:
                    # 如果是核心点，将其邻居加入种子列表
                    for neighbor_idx in neighbors:
                        if neighbor_idx not in seeds and labels[neighbor_idx] <= 0:
                            seeds.append(neighbor_idx)

            i += 1

    def _compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        计算两个点之间的距离

        Args:
            point1: 第一个点 [lat, lon]
            point2: 第二个点 [lat, lon]

        Returns:
            两点之间的距离（米）
        """
        if self.metric == 'euclidean':
            # 简化版欧氏距离（适用于小范围区域）
            return np.sqrt(np.sum((point1 - point2) ** 2))

        elif self.metric == 'haversine':
            # Haversine距离，适用于地理坐标
            from math import radians, sin, cos, sqrt, atan2

            # 将十进制度数转化为弧度
            lat1, lon1 = radians(point1[0]), radians(point1[1])
            lat2, lon2 = radians(point2[0]), radians(point2[1])

            # Haversine公式
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            # 地球平均半径（米）
            R = 6371000
            return R * c

        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")

    def get_cluster_stats(self) -> dict:
        """
        获取聚类统计信息

        Returns:
            包含聚类统计信息的字典
        """
        if self.labels_ is None:
            return {}

        unique_labels = np.unique(self.labels_)
        stats = {
            'n_clusters': len(unique_labels) - (1 if 0 in unique_labels else 0),
            'n_noise': np.sum(self.labels_ == 0),
            'n_core_points': len(self.core_sample_indices_),
            'execution_time': self.execution_time,
            'cluster_sizes': {}
        }

        for label in unique_labels:
            if label != 0:  # 跳过噪声点
                stats['cluster_sizes'][label] = np.sum(self.labels_ == label)

        return stats