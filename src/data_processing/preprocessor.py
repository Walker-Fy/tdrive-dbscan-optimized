"""
轨迹数据预处理器
数据清洗、采样和特征提取
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import euclidean
import warnings
from math import radians, sin, cos, sqrt, atan2

from .trajectory import Trajectory, TrajectoryPoint


class TrajectoryPreprocessor:
    """轨迹数据预处理器"""

    def __init__(self,
                 min_trajectory_length: int = 10,
                 max_speed_kmh: float = 200.0,
                 max_acceleration_ms2: float = 10.0,
                 sampling_interval_s: float = 60.0,
                 remove_outliers: bool = True):
        """
        初始化预处理器参数

        Args:
            min_trajectory_length: 最小轨迹长度（点数）
            max_speed_kmh: 最大合理速度（km/h）
            max_acceleration_ms2: 最大合理加速度（m/s²）
            sampling_interval_s: 重采样间隔（秒）
            remove_outliers: 是否移除异常值
        """
        self.min_trajectory_length = min_trajectory_length
        self.max_speed_kmh = max_speed_kmh
        self.max_acceleration_ms2 = max_acceleration_ms2
        self.sampling_interval_s = sampling_interval_s
        self.remove_outliers = remove_outliers

        # 北京的大致边界坐标
        self.beijing_bounds = {
            'min_lat': 39.4,  # 南边界
            'max_lat': 40.6,  # 北边界
            'min_lon': 115.7,  # 西边界
            'max_lon': 117.4  # 东边界
        }

    def preprocess_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        预处理轨迹列表

        Args:
            trajectories: 原始轨迹列表

        Returns:
            预处理后的轨迹列表
        """
        processed_trajectories = []

        from tqdm import tqdm
        for traj in tqdm(trajectories):
            try:
                processed_traj = self._preprocess_single_trajectory(traj)
                if processed_traj is not None:
                    processed_trajectories.append(processed_traj)
            except Exception as e:
                warnings.warn(f"预处理轨迹 {traj.taxi_id} 时出错: {e}")
                continue

        print(f"预处理完成: {len(processed_trajectories)}/{len(trajectories)} 条轨迹保留")
        return processed_trajectories

    def _preprocess_single_trajectory(self, trajectory: Trajectory) -> Optional[Trajectory]:
        """
        预处理单条轨迹

        Args:
            trajectory: 原始轨迹

        Returns:
            预处理后的轨迹，如果无效则返回None
        """
        # 1. 移除无效点
        clean_points = self._remove_invalid_points(trajectory.points)

        if len(clean_points) < self.min_trajectory_length:
            return None

        # 2. 按时间排序
        sorted_points = sorted(clean_points, key=lambda p: p.timestamp)

        # 3. 计算速度和方向
        points_with_features = self._compute_movement_features(sorted_points)

        # 4. 移除速度异常的点
        if self.remove_outliers:
            filtered_points = self._filter_outliers(points_with_features)
            if len(filtered_points) < self.min_trajectory_length:
                return None
        else:
            filtered_points = points_with_features

        # 5. 重采样（等时间间隔）
        resampled_points = self._resample_trajectory(filtered_points)

        if len(resampled_points) < self.min_trajectory_length:
            return None

        # 6. 创建新的轨迹对象
        processed_trajectory = Trajectory(
            taxi_id=trajectory.taxi_id,
            points=resampled_points,
            metadata=trajectory.metadata.copy() if trajectory.metadata else {}
        )

        # 7. 计算轨迹统计信息
        self._compute_trajectory_stats(processed_trajectory)

        return processed_trajectory

    def _remove_invalid_points(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        移除无效的轨迹点

        Args:
            points: 轨迹点列表

        Returns:
            有效的轨迹点列表
        """
        valid_points = []

        for point in points:
            # 检查坐标是否有效
            if (pd.isna(point.latitude) or pd.isna(point.longitude) or
                    pd.isna(point.timestamp)):
                continue

            # 检查坐标是否在北京范围内
            if not self._is_in_beijing(point.latitude, point.longitude):
                continue

            # 检查时间戳是否合理
            if point.timestamp.year < 2000 or point.timestamp.year > 2030:
                continue

            valid_points.append(point)

        return valid_points

    def _is_in_beijing(self, lat: float, lon: float) -> bool:
        """
        检查坐标是否在北京范围内

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            如果在北京范围内返回True
        """
        bounds = self.beijing_bounds
        return (bounds['min_lat'] <= lat <= bounds['max_lat'] and
                bounds['min_lon'] <= lon <= bounds['max_lon'])

    def _compute_movement_features(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        计算轨迹点的运动特征（速度、方向）

        Args:
            points: 按时间排序的轨迹点

        Returns:
            添加了特征的轨迹点列表
        """
        if len(points) < 2:
            return points

        # 初始化第一个点
        points[0].speed = 0.0
        points[0].direction = 0.0
        points[0].acceleration = 0.0

        for i in range(1, len(points)):
            prev_point = points[i - 1]
            curr_point = points[i]

            # 计算时间差（秒）
            time_diff = (curr_point.timestamp - prev_point.timestamp).total_seconds()
            if time_diff <= 0:
                time_diff = 1.0  # 避免除零

            # 计算距离（米）
            distance = self._haversine_distance(
                prev_point.latitude, prev_point.longitude,
                curr_point.latitude, curr_point.longitude
            )

            # 计算速度（m/s）
            speed = distance / time_diff

            # 转换为 km/h
            speed_kmh = speed * 3.6

            # 计算方向（角度，0-360，正北为0）
            direction = self._compute_direction(
                prev_point.latitude, prev_point.longitude,
                curr_point.latitude, curr_point.longitude
            )

            # 计算加速度（m/s²）
            if i >= 2:
                prev_speed = points[i - 1].speed
                acceleration = (speed - prev_speed) / time_diff if time_diff > 0 else 0.0
            else:
                acceleration = 0.0

            # 更新当前点的特征
            curr_point.speed = speed
            curr_point.speed_kmh = speed_kmh
            curr_point.direction = direction
            curr_point.acceleration = acceleration
            curr_point.distance_from_prev = distance
            curr_point.time_from_prev = time_diff

        return points

    def _haversine_distance(self, lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """
        使用Haversine公式计算两点间的大圆距离

        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度

        Returns:
            距离（米）
        """
        R = 6371000.0  # 地球半径（米）

        # 转换为弧度
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine公式
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def _compute_direction(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """
        计算从点1到点2的方向（角度）

        Args:
            lat1, lon1: 起点坐标
            lat2, lon2: 终点坐标

        Returns:
            方向角度（0-360度，正北为0）
        """
        # 转换为弧度
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)

        # 计算经度差
        dlon = lon2_rad - lon1_rad

        # 计算方向
        y = sin(dlon) * cos(lat2_rad)
        x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)

        # 计算方位角（弧度）
        bearing = atan2(y, x)

        # 转换为角度并标准化到0-360
        bearing_deg = np.degrees(bearing)
        bearing_deg = (bearing_deg + 360) % 360

        return bearing_deg

    def _filter_outliers(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        基于速度和加速度过滤异常点

        Args:
            points: 轨迹点列表

        Returns:
            过滤后的轨迹点列表
        """
        if len(points) < 3:
            return points

        valid_indices = []

        for i, point in enumerate(points):
            # 跳过第一个点（没有前一速度）
            if i == 0:
                valid_indices.append(i)
                continue

            # 检查速度是否合理
            speed_kmh = getattr(point, 'speed_kmh', 0.0)
            if speed_kmh > self.max_speed_kmh:
                continue

            # 检查加速度是否合理
            acceleration = getattr(point, 'acceleration', 0.0)
            if abs(acceleration) > self.max_acceleration_ms2:
                continue

            valid_indices.append(i)

        # 返回过滤后的点
        return [points[i] for i in valid_indices]

    def _resample_trajectory(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        对轨迹进行等时间间隔重采样

        Args:
            points: 原始轨迹点（按时间排序）

        Returns:
            重采样后的轨迹点
        """
        if len(points) < 2:
            return points

        # 获取时间范围
        start_time = points[0].timestamp
        end_time = points[-1].timestamp

        # 创建重采样时间点
        resampled_times = []
        current_time = start_time
        interval = pd.Timedelta(seconds=self.sampling_interval_s)

        while current_time <= end_time:
            resampled_times.append(current_time)
            current_time += interval

        if not resampled_times:
            return points

        # 线性插值获取重采样点
        resampled_points = []

        # 提取原始数据
        times = [p.timestamp for p in points]
        lats = [p.latitude for p in points]
        lons = [p.longitude for p in points]

        for target_time in resampled_times:
            # 找到最近的原始点
            closest_idx = np.argmin([abs((t - target_time).total_seconds()) for t in times])

            # 创建新的轨迹点
            closest_point = points[closest_idx]
            new_point = TrajectoryPoint(
                latitude=closest_point.latitude,
                longitude=closest_point.longitude,
                timestamp=target_time,
                taxi_id=closest_point.taxi_id,
                speed=getattr(closest_point, 'speed', 0.0),
                direction=getattr(closest_point, 'direction', 0.0)
            )

            resampled_points.append(new_point)

        return resampled_points

    def _compute_trajectory_stats(self, trajectory: Trajectory) -> None:
        """
        计算轨迹统计信息并存储到metadata中

        Args:
            trajectory: 轨迹对象
        """
        if len(trajectory.points) < 2:
            return

        # 计算总距离
        total_distance = 0.0
        speeds = []

        for i in range(1, len(trajectory.points)):
            p1 = trajectory.points[i - 1]
            p2 = trajectory.points[i]

            distance = self._haversine_distance(
                p1.latitude, p1.longitude,
                p2.latitude, p2.longitude
            )
            total_distance += distance

            speed = getattr(p2, 'speed', 0.0)
            speeds.append(speed)

        # 计算持续时间
        start_time = trajectory.points[0].timestamp
        end_time = trajectory.points[-1].timestamp
        duration_hours = (end_time - start_time).total_seconds() / 3600.0

        # 平均速度
        avg_speed = np.mean(speeds) if speeds else 0.0

        # 更新metadata
        trajectory.metadata.update({
            'total_distance_km': total_distance / 1000.0,
            'duration_hours': duration_hours,
            'avg_speed_kmh': avg_speed * 3.6,
            'n_points': len(trajectory.points),
            'start_time': start_time,
            'end_time': end_time
        })


def clean_trajectory_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    快速清洗轨迹数据（DataFrame版本）

    Args:
        df: 原始轨迹DataFrame

    Returns:
        清洗后的DataFrame
    """
    # 移除缺失值
    df_clean = df.dropna(subset=['latitude', 'longitude', 'timestamp'])

    # 移除重复行
    df_clean = df_clean.drop_duplicates()

    # 移除不合理坐标
    beijing_bounds = {
        'min_lat': 39.4, 'max_lat': 40.6,
        'min_lon': 115.7, 'max_lon': 117.4
    }

    mask = (
            (df_clean['latitude'] >= beijing_bounds['min_lat']) &
            (df_clean['latitude'] <= beijing_bounds['max_lat']) &
            (df_clean['longitude'] >= beijing_bounds['min_lon']) &
            (df_clean['longitude'] <= beijing_bounds['max_lon'])
    )

    df_clean = df_clean[mask]

    return df_clean


def resample_trajectories(trajectories: List[Trajectory],
                          interval_minutes: float = 1.0) -> List[Trajectory]:
    """
    重采样轨迹集合

    Args:
        trajectories: 原始轨迹列表
        interval_minutes: 重采样间隔（分钟）

    Returns:
        重采样后的轨迹列表
    """
    preprocessor = TrajectoryPreprocessor(sampling_interval_s=interval_minutes * 60)
    return preprocessor.preprocess_trajectories(trajectories)