"""
聚类工具函数
提供DBSCAN算法中的通用函数和优化工具
"""

import numpy as np
from typing import List, Tuple, Dict, Set
import math
from numba import jit, prange
import warnings

try:
    from scipy.spatial import KDTree

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. KDTree optimizations disabled.")


def compute_distance_matrix(points: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    计算距离矩阵（优化版本）

    Args:
        points: 形状为(n_samples, 2)的numpy数组
        metric: 距离度量方式

    Returns:
        距离矩阵，形状为(n_samples, n_samples)
    """
    n_samples = points.shape[0]

    if metric == 'euclidean':
        # 使用向量化计算欧氏距离
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        return distance_matrix

    elif metric == 'haversine':
        # 计算Haversine距离矩阵
        return _haversine_distance_matrix(points)

    else:
        raise ValueError(f"不支持的度量方式: {metric}")


@jit(nopython=True, parallel=True)
def _haversine_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    使用Numba加速的Haversine距离矩阵计算

    Args:
        points: 形状为(n_samples, 2)的numpy数组，[latitude, longitude]

    Returns:
        Haversine距离矩阵
    """
    n_samples = points.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    R = 6371000.0  # 地球半径（米）

    # 转换为弧度
    lat_rad = np.radians(points[:, 0])
    lon_rad = np.radians(points[:, 1])

    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            dlon = lon_rad[j] - lon_rad[i]
            dlat = lat_rad[j] - lat_rad[i]

            # Haversine公式
            a = math.sin(dlat / 2) ** 2 + math.cos(lat_rad[i]) * math.cos(lat_rad[j]) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def region_query(points: np.ndarray, point_idx: int, eps: float,
                 distance_matrix: np.ndarray = None,
                 tree: KDTree = None) -> List[int]:
    """
    查找指定点邻域内的所有点（优化版本）

    Args:
        points: 所有点的数组
        point_idx: 目标点的索引
        eps: 邻域半径
        distance_matrix: 预计算的距离矩阵（可选）
        tree: KDTree索引（可选）

    Returns:
        邻域内点的索引列表
    """
    n_samples = points.shape[0]

    if tree is not None and SCIPY_AVAILABLE:
        # 使用KDTree进行快速区域查询
        neighbors = tree.query_ball_point(points[point_idx], eps)
        return neighbors

    elif distance_matrix is not None:
        # 使用预计算的距离矩阵
        neighbors = np.where(distance_matrix[point_idx] <= eps)[0].tolist()
        neighbors.remove(point_idx)  # 移除自身
        return neighbors

    else:
        # 普通计算方法
        neighbors = []
        point = points[point_idx]

        for i in range(n_samples):
            if i == point_idx:
                continue

            distance = np.linalg.norm(point - points[i])
            if distance <= eps:
                neighbors.append(i)

        return neighbors


def merge_clusters(labels_list: List[np.ndarray], overlap_threshold: float = 0.5) -> np.ndarray:
    """
    合并多个聚类结果

    Args:
        labels_list: 多个聚类标签列表
        overlap_threshold: 重叠阈值，用于决定是否合并聚类

    Returns:
        合并后的聚类标签
    """
    if not labels_list:
        return np.array([])

    # 以第一个聚类结果为基础
    base_labels = labels_list[0]
    merged_labels = base_labels.copy()

    for other_labels in labels_list[1:]:
        # 对每个聚类进行合并
        merged_labels = _merge_two_clusters(merged_labels, other_labels, overlap_threshold)

    return merged_labels


def _merge_two_clusters(labels1: np.ndarray, labels2: np.ndarray,
                        threshold: float) -> np.ndarray:
    """
    合并两个聚类结果

    Args:
        labels1: 第一个聚类标签
        labels2: 第二个聚类标签
        threshold: 重叠阈值

    Returns:
        合并后的聚类标签
    """
    n_samples = len(labels1)
    merged_labels = labels1.copy()

    # 创建标签映射
    label_map = {}

    for label in np.unique(labels2):
        if label == 0:  # 跳过噪声点
            continue

        # 找到labels2中该聚类对应的点
        mask = labels2 == label
        points_in_cluster = np.where(mask)[0]

        # 查找与labels1中的聚类重叠
        overlapping_labels = labels1[mask]
        unique_overlapping, counts = np.unique(
            overlapping_labels[overlapping_labels > 0],
            return_counts=True
        )

        if len(unique_overlapping) == 0:
            # 新的聚类
            new_label = np.max(merged_labels) + 1
            merged_labels[mask] = new_label
        else:
            # 找到重叠最多的聚类
            max_overlap_idx = np.argmax(counts)
            max_overlap_label = unique_overlapping[max_overlap_idx]
            overlap_ratio = counts[max_overlap_idx] / len(points_in_cluster)

            if overlap_ratio >= threshold:
                # 合并到现有聚类
                merged_labels[mask] = max_overlap_label
            else:
                # 创建新聚类
                new_label = np.max(merged_labels) + 1
                merged_labels[mask] = new_label

    return merged_labels


def build_spatial_index(points: np.ndarray, method: str = 'kdtree') -> object:
    """
    构建空间索引以加速邻域查询

    Args:
        points: 点数据数组
        method: 索引方法，支持'kdtree'或'balltree'

    Returns:
        空间索引对象
    """
    if method == 'kdtree' and SCIPY_AVAILABLE:
        return KDTree(points)

    elif method == 'balltree':
        try:
            from sklearn.neighbors import BallTree
            return BallTree(points, metric='haversine')
        except ImportError:
            warnings.warn("scikit-learn not available. Cannot build BallTree.")
            return None

    else:
        warnings.warn(f"Unsupported spatial index method: {method}")
        return None


def optimize_parameters(points: np.ndarray, eps_range: List[float],
                        min_samples_range: List[int]) -> Dict:
    """
    通过网格搜索优化DBSCAN参数

    Args:
        points: 点数据
        eps_range: eps参数范围
        min_samples_range: min_samples参数范围

    Returns:
        最优参数和相应的聚类质量
    """
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                # 这里可以添加聚类质量评估
                # 例如：轮廓系数、Davies-Bouldin指数等
                n_clusters_estimated = _estimate_n_clusters(points, eps, min_samples)

                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters_estimated,
                    'score': n_clusters_estimated  # 简化评分
                })
            except Exception as e:
                print(f"参数 eps={eps}, min_samples={min_samples} 失败: {e}")

    if not results:
        return {'eps': 100.0, 'min_samples': 5, 'score': 0}

    # 选择评分最高的参数
    best_result = max(results, key=lambda x: x['score'])
    return best_result


def _estimate_n_clusters(points: np.ndarray, eps: float, min_samples: int) -> int:
    """
    粗略估计聚类数量（用于参数优化）

    Args:
        points: 点数据
        eps: 邻域半径
        min_samples: 最小样本数

    Returns:
        估计的聚类数量
    """
    from .dbscan_sequential import DBSCANSequential

    # 使用小样本进行快速估计
    n_samples = min(1000, len(points))
    indices = np.random.choice(len(points), n_samples, replace=False)
    sample_points = points[indices]

    dbscan = DBSCANSequential(eps=eps, min_samples=min_samples)
    dbscan.fit(sample_points)

    n_clusters = len(np.unique(dbscan.labels_)) - (1 if 0 in dbscan.labels_ else 0)
    return n_clusters