"""
数据分区策略
将轨迹数据划分为可并行处理的块
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.neighbors import KDTree
import multiprocessing as mp
from dataclasses import dataclass
import warnings


@dataclass
class Partition:
    """数据分区信息"""
    id: int
    indices: np.ndarray
    points: np.ndarray
    bounds: Dict[str, float]
    centroid: Tuple[float, float]
    n_points: int = 0

    def __post_init__(self):
        self.n_points = len(self.indices)


class DataPartitioner:
    """数据分区器基类"""

    def __init__(self, n_partitions: int, overlap_ratio: float = 0.1):
        """
        初始化分区器

        Args:
            n_partitions: 分区数量
            overlap_ratio: 分区重叠比例（用于处理边界点）
        """
        self.n_partitions = n_partitions
        self.overlap_ratio = overlap_ratio
        self.partitions: List[Partition] = []

    def partition(self, points: np.ndarray) -> List[Partition]:
        """
        分区方法（子类需实现）

        Args:
            points: 点数据数组

        Returns:
            分区列表
        """
        raise NotImplementedError

    def _calculate_bounds(self, points: np.ndarray) -> Dict[str, float]:
        """
        计算点的边界

        Args:
            points: 点数据

        Returns:
            边界字典
        """
        return {
            'min_x': np.min(points[:, 0]),
            'max_x': np.max(points[:, 0]),
            'min_y': np.min(points[:, 1]),
            'max_y': np.max(points[:, 1])
        }

    def _calculate_centroid(self, points: np.ndarray) -> Tuple[float, float]:
        """
        计算点的质心

        Args:
            points: 点数据

        Returns:
            (x, y) 质心坐标
        """
        return np.mean(points[:, 0]), np.mean(points[:, 1])

    def get_partition_info(self) -> Dict[str, Any]:
        """
        获取分区统计信息

        Returns:
            分区信息字典
        """
        if not self.partitions:
            return {}

        n_points_list = [p.n_points for p in self.partitions]

        return {
            'n_partitions': len(self.partitions),
            'total_points': sum(n_points_list),
            'min_points_per_partition': min(n_points_list),
            'max_points_per_partition': max(n_points_list),
            'avg_points_per_partition': np.mean(n_points_list),
            'std_points_per_partition': np.std(n_points_list),
            'partition_method': self.__class__.__name__
        }


class GridPartitioner(DataPartitioner):
    """基于网格的空间分区器"""

    def __init__(self, n_partitions: int, overlap_ratio: float = 0.1):
        super().__init__(n_partitions, overlap_ratio)

    def partition(self, points: np.ndarray) -> List[Partition]:
        """
        使用网格方法进行分区

        Args:
            points: 点数据数组

        Returns:
            分区列表
        """
        n_points = points.shape[0]

        # 计算网格维度
        grid_size = int(np.sqrt(self.n_partitions))
        if grid_size * grid_size != self.n_partitions:
            warnings.warn(f"分区数 {self.n_partitions} 不是完全平方数，使用 {grid_size * grid_size} 个分区")

        # 计算边界
        bounds = self._calculate_bounds(points)
        x_range = bounds['max_x'] - bounds['min_x']
        y_range = bounds['max_y'] - bounds['min_y']

        # 计算网格边界
        x_step = x_range / grid_size
        y_step = y_range / grid_size

        # 重叠边界扩展
        x_overlap = x_step * self.overlap_ratio
        y_overlap = y_step * self.overlap_ratio

        partitions = []
        partition_id = 0

        # 为每个网格单元创建分区
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算网格边界（带重叠）
                x_min = bounds['min_x'] + i * x_step - x_overlap
                x_max = bounds['min_x'] + (i + 1) * x_step + x_overlap
                y_min = bounds['min_y'] + j * y_step - y_overlap
                y_max = bounds['min_y'] + (j + 1) * y_step + y_overlap

                # 查找在边界内的点
                mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                        (points[:, 1] >= y_min) & (points[:, 1] <= y_max))

                indices = np.where(mask)[0]

                if len(indices) > 0:
                    partition_points = points[indices]

                    partition = Partition(
                        id=partition_id,
                        indices=indices,
                        points=partition_points,
                        bounds={'min_x': x_min, 'max_x': x_max,
                                'min_y': y_min, 'max_y': y_max},
                        centroid=self._calculate_centroid(partition_points)
                    )
                    partitions.append(partition)
                    partition_id += 1

        self.partitions = partitions
        return partitions

    def get_grid_info(self) -> Dict[str, Any]:
        """
        获取网格分区详细信息

        Returns:
            网格信息字典
        """
        if not self.partitions:
            return {}

        # 计算网格维度
        grid_size = int(np.sqrt(len(self.partitions)))

        return {
            'grid_size': grid_size,
            'n_cells': grid_size * grid_size,
            'effective_partitions': len(self.partitions)
        }


class KDTreePartitioner(DataPartitioner):
    """基于KD树的分区器（实现负载均衡）"""

    def __init__(self, n_partitions: int, overlap_ratio: float = 0.1,
                 balance_tolerance: float = 0.2):
        """
        初始化KD树分区器

        Args:
            n_partitions: 分区数量
            overlap_ratio: 重叠比例
            balance_tolerance: 负载均衡容忍度
        """
        super().__init__(n_partitions, overlap_ratio)
        self.balance_tolerance = balance_tolerance

    def partition(self, points: np.ndarray) -> List[Partition]:
        """
        使用KD树进行分区

        Args:
            points: 点数据数组

        Returns:
            分区列表
        """
        n_points = points.shape[0]

        # 构建KD树
        kdtree = KDTree(points)

        # 使用递归划分实现负载均衡
        partitions = self._recursive_partition(
            points, kdtree,
            indices=np.arange(n_points),
            depth=0,
            max_depth=int(np.log2(self.n_partitions)) + 2
        )

        # 创建分区对象
        self.partitions = []
        for i, (indices, partition_points) in enumerate(partitions):
            if len(indices) > 0:
                partition = Partition(
                    id=i,
                    indices=indices,
                    points=partition_points,
                    bounds=self._calculate_bounds(partition_points),
                    centroid=self._calculate_centroid(partition_points)
                )
                self.partitions.append(partition)

        return self.partitions

    def _recursive_partition(self, points: np.ndarray, kdtree: KDTree,
                             indices: np.ndarray, depth: int, max_depth: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        递归划分数据

        Args:
            points: 所有点数据
            kdtree: KD树
            indices: 当前节点的点索引
            depth: 当前递归深度
            max_depth: 最大递归深度

        Returns:
            分区列表
        """
        n_points = len(indices)

        # 终止条件
        if depth >= max_depth or n_points <= 1:
            return [(indices, points[indices])]

        # 计算当前节点的质心
        current_points = points[indices]
        centroid = self._calculate_centroid(current_points)

        # 找到距离质心最近的点作为分割点
        dist, nearest_idx = kdtree.query([centroid], k=1)
        split_point_idx = indices[nearest_idx[0][0]]
        split_point = points[split_point_idx]

        # 计算与分割点的距离
        distances = np.linalg.norm(current_points - split_point, axis=1)
        median_distance = np.median(distances)

        # 基于距离划分点
        left_mask = distances <= median_distance
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        # 检查是否达到负载均衡
        left_ratio = len(left_indices) / n_points
        if (0.5 - self.balance_tolerance) <= left_ratio <= (0.5 + self.balance_tolerance):
            # 均衡划分，继续递归
            left_partitions = self._recursive_partition(
                points, kdtree, left_indices, depth + 1, max_depth
            )
            right_partitions = self._recursive_partition(
                points, kdtree, right_indices, depth + 1, max_depth
            )
            return left_partitions + right_partitions
        else:
            # 不均衡，不再划分
            return [(indices, current_points)]

    def _balance_partitions(self, partitions: List[Partition]) -> List[Partition]:
        """
        平衡分区大小

        Args:
            partitions: 原始分区列表

        Returns:
            平衡后的分区列表
        """
        if not partitions:
            return []

        # 计算目标大小
        total_points = sum(p.n_points for p in partitions)
        target_size = total_points / len(partitions)

        # 平衡分区
        balanced_partitions = []
        current_partition = None

        for partition in partitions:
            if current_partition is None:
                current_partition = partition
            elif current_partition.n_points < target_size * (1 - self.balance_tolerance):
                # 合并小分区
                current_partition = self._merge_partitions(current_partition, partition)
            else:
                balanced_partitions.append(current_partition)
                current_partition = partition

        if current_partition is not None:
            balanced_partitions.append(current_partition)

        return balanced_partitions

    def _merge_partitions(self, p1: Partition, p2: Partition) -> Partition:
        """
        合并两个分区

        Args:
            p1: 第一个分区
            p2: 第二个分区

        Returns:
            合并后的分区
        """
        merged_indices = np.concatenate([p1.indices, p2.indices])
        merged_points = np.vstack([p1.points, p2.points])

        return Partition(
            id=p1.id,
            indices=merged_indices,
            points=merged_points,
            bounds=self._calculate_merged_bounds(p1.bounds, p2.bounds),
            centroid=self._calculate_centroid(merged_points)
        )

    def _calculate_merged_bounds(self, bounds1: Dict, bounds2: Dict) -> Dict:
        """
        计算合并后的边界

        Args:
            bounds1: 第一个边界
            bounds2: 第二个边界

        Returns:
            合并后的边界
        """
        return {
            'min_x': min(bounds1['min_x'], bounds2['min_x']),
            'max_x': max(bounds1['max_x'], bounds2['max_x']),
            'min_y': min(bounds1['min_y'], bounds2['min_y']),
            'max_y': max(bounds1['max_y'], bounds2['max_y'])
        }


class SpatialHashPartitioner(DataPartitioner):
    """基于空间哈希的分区器"""

    def __init__(self, n_partitions: int, overlap_ratio: float = 0.1,
                 cell_size: Optional[float] = None):
        """
        初始化空间哈希分区器

        Args:
            n_partitions: 分区数量
            overlap_ratio: 重叠比例
            cell_size: 哈希单元大小（如果为None则自动计算）
        """
        super().__init__(n_partitions, overlap_ratio)
        self.cell_size = cell_size

    def partition(self, points: np.ndarray) -> List[Partition]:
        """
        使用空间哈希进行分区

        Args:
            points: 点数据数组

        Returns:
            分区列表
        """
        bounds = self._calculate_bounds(points)
        x_range = bounds['max_x'] - bounds['min_x']
        y_range = bounds['max_y'] - bounds['min_y']

        # 计算哈希单元大小
        if self.cell_size is None:
            # 自动计算单元大小以实现目标分区数
            area = x_range * y_range
            self.cell_size = np.sqrt(area / self.n_partitions)

        # 计算网格维度
        n_cells_x = int(np.ceil(x_range / self.cell_size))
        n_cells_y = int(np.ceil(y_range / self.cell_size))

        # 创建哈希映射
        cell_map = {}

        for idx, point in enumerate(points):
            # 计算哈希键
            cell_x = int((point[0] - bounds['min_x']) / self.cell_size)
            cell_y = int((point[1] - bounds['min_y']) / self.cell_size)
            cell_key = (cell_x, cell_y)

            if cell_key not in cell_map:
                cell_map[cell_key] = []
            cell_map[cell_key].append(idx)

        # 创建分区
        partitions = []
        for partition_id, (cell_key, indices) in enumerate(cell_map.items()):
            partition_points = points[indices]

            # 计算边界（带重叠）
            x_min = bounds['min_x'] + cell_key[0] * self.cell_size - self.cell_size * self.overlap_ratio
            x_max = x_min + self.cell_size + 2 * self.cell_size * self.overlap_ratio
            y_min = bounds['min_y'] + cell_key[1] * self.cell_size - self.cell_size * self.overlap_ratio
            y_max = y_min + self.cell_size + 2 * self.cell_size * self.overlap_ratio

            partition = Partition(
                id=partition_id,
                indices=np.array(indices),
                points=partition_points,
                bounds={'min_x': x_min, 'max_x': x_max,
                        'min_y': y_min, 'max_y': y_max},
                centroid=self._calculate_centroid(partition_points)
            )
            partitions.append(partition)

        self.partitions = partitions
        return partitions