"""
轨迹数据结构定义
轨迹点、轨迹段和完整轨迹的数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt


@dataclass
class TrajectoryPoint:
    """轨迹点数据结构"""

    latitude: float  # 纬度
    longitude: float  # 经度
    timestamp: pd.Timestamp  # 时间戳
    taxi_id: str  # 出租车ID

    # 计算得到的特征
    speed: Optional[float] = None  # 速度 (m/s)
    speed_kmh: Optional[float] = None  # 速度 (km/h)
    direction: Optional[float] = None  # 方向角度 (0-360度)
    acceleration: Optional[float] = None  # 加速度 (m/s²)
    distance_from_prev: Optional[float] = None  # 与上一个点的距离 (米)
    time_from_prev: Optional[float] = None  # 与上一个点的时间差 (秒)

    # 其他属性
    elevation: Optional[float] = None  # 海拔
    accuracy: Optional[float] = None  # GPS精度
    is_stop: bool = False  # 是否是停留点

    def to_array(self) -> np.ndarray:
        """
        将轨迹点转换为numpy数组

        Returns:
            [latitude, longitude] 数组
        """
        return np.array([self.latitude, self.longitude])

    def to_dict(self) -> Dict[str, Any]:
        """
        将轨迹点转换为字典

        Returns:
            包含所有属性的字典
        """
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timestamp': self.timestamp,
            'taxi_id': self.taxi_id,
            'speed': self.speed,
            'speed_kmh': self.speed_kmh,
            'direction': self.direction,
            'acceleration': self.acceleration,
            'distance_from_prev': self.distance_from_prev,
            'time_from_prev': self.time_from_prev,
            'elevation': self.elevation,
            'accuracy': self.accuracy,
            'is_stop': self.is_stop
        }

    def distance_to(self, other: 'TrajectoryPoint') -> float:
        """
        计算到另一个点的距离（使用Haversine公式）

        Args:
            other: 另一个轨迹点

        Returns:
            距离（米）
        """
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000.0  # 地球半径（米）

        # 转换为弧度
        lat1, lon1 = radians(self.latitude), radians(self.longitude)
        lat2, lon2 = radians(other.latitude), radians(other.longitude)

        # 计算差值
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine公式
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def time_difference(self, other: 'TrajectoryPoint') -> float:
        """
        计算与另一个点的时间差

        Args:
            other: 另一个轨迹点

        Returns:
            时间差（秒）
        """
        return abs((self.timestamp - other.timestamp).total_seconds())


@dataclass
class TrajectorySegment:
    """轨迹段数据结构（连续的轨迹点序列）"""

    points: List[TrajectoryPoint]
    segment_id: str
    parent_taxi_id: str

    # 段属性
    start_time: pd.Timestamp = field(init=False)
    end_time: pd.Timestamp = field(init=False)
    length_meters: float = field(init=False)
    duration_seconds: float = field(init=False)

    def __post_init__(self):
        """初始化后计算派生属性"""
        if not self.points:
            raise ValueError("轨迹段不能为空")

        self.start_time = self.points[0].timestamp
        self.end_time = self.points[-1].timestamp
        self.length_meters = self._calculate_length()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def _calculate_length(self) -> float:
        """计算轨迹段总长度"""
        total_length = 0.0

        for i in range(1, len(self.points)):
            total_length += self.points[i].distance_to(self.points[i - 1])

        return total_length

    def get_bounding_box(self) -> Dict[str, float]:
        """
        获取轨迹段的边界框

        Returns:
            包含min/max纬度和经度的字典
        """
        lats = [p.latitude for p in self.points]
        lons = [p.longitude for p in self.points]

        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }

    def get_centroid(self):
        """
        获取轨迹段的质心

        Returns:
            (纬度, 经度) 元组
        """
        lats = [p.latitude for p in self.points]
        lons = [p.longitude for p in self.points]

        return np.mean(lats), np.mean(lons)

    def to_dataframe(self) -> pd.DataFrame:
        """
        将轨迹段转换为DataFrame

        Returns:
            包含所有点的DataFrame
        """
        data = [point.to_dict() for point in self.points]
        return pd.DataFrame(data)


class Trajectory:
    """完整轨迹数据结构"""

    def __init__(self, taxi_id: str, points: List[TrajectoryPoint],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        初始化轨迹

        Args:
            taxi_id: 出租车ID
            points: 轨迹点列表
            metadata: 轨迹元数据
        """
        self.taxi_id = taxi_id
        self.points = points
        self.metadata = metadata or {}

        # 计算轨迹统计信息
        self._update_stats()

    def _update_stats(self) -> None:
        """更新轨迹统计信息"""
        if not self.points:
            return

        # 基本信息
        self.start_time = self.points[0].timestamp
        self.end_time = self.points[-1].timestamp
        self.duration = (self.end_time - self.start_time).total_seconds()

        # 计算总距离
        self.total_distance = 0.0
        speeds = []

        for i in range(1, len(self.points)):
            p1 = self.points[i - 1]
            p2 = self.points[i]

            distance = p2.distance_to(p1)
            self.total_distance += distance

            if hasattr(p2, 'speed') and p2.speed is not None:
                speeds.append(p2.speed)

        # 平均速度
        self.avg_speed = np.mean(speeds) if speeds else 0.0
        self.max_speed = max(speeds) if speeds else 0.0

        # 点密度
        self.point_density = len(self.points) / self.total_distance if self.total_distance > 0 else 0

        # 更新metadata
        self.metadata.update({
            'total_distance_km': self.total_distance / 1000.0,
            'duration_hours': self.duration / 3600.0,
            'avg_speed_kmh': self.avg_speed * 3.6,
            'max_speed_kmh': self.max_speed * 3.6,
            'n_points': len(self.points),
            'point_density_per_km': self.point_density * 1000
        })

    def get_segments(self, max_gap_minutes: float = 30.0) -> List[TrajectorySegment]:
        """
        将轨迹分割为多个段（基于时间间隙）

        Args:
            max_gap_minutes: 最大允许的时间间隙（分钟）

        Returns:
            轨迹段列表
        """
        if len(self.points) < 2:
            return []

        segments = []
        current_segment = [self.points[0]]

        for i in range(1, len(self.points)):
            prev_point = self.points[i - 1]
            curr_point = self.points[i]

            # 计算时间差
            time_gap = (curr_point.timestamp - prev_point.timestamp).total_seconds() / 60.0

            if time_gap > max_gap_minutes:
                # 时间间隙过大，结束当前段
                if len(current_segment) > 1:
                    segment_id = f"{self.taxi_id}_segment_{len(segments)}"
                    segment = TrajectorySegment(
                        points=current_segment.copy(),
                        segment_id=segment_id,
                        parent_taxi_id=self.taxi_id
                    )
                    segments.append(segment)

                # 开始新段
                current_segment = [curr_point]
            else:
                current_segment.append(curr_point)

        # 添加最后一个段
        if len(current_segment) > 1:
            segment_id = f"{self.taxi_id}_segment_{len(segments)}"
            segment = TrajectorySegment(
                points=current_segment,
                segment_id=segment_id,
                parent_taxi_id=self.taxi_id
            )
            segments.append(segment)

        return segments

    def get_bounding_box(self) -> Dict[str, float]:
        """
        获取轨迹的边界框

        Returns:
            包含min/max纬度和经度的字典
        """
        lats = [p.latitude for p in self.points]
        lons = [p.longitude for p in self.points]

        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }

    def get_centroid(self):
        """
        获取轨迹的质心

        Returns:
            (纬度, 经度) 元组
        """
        lats = [p.latitude for p in self.points]
        lons = [p.longitude for p in self.points]

        return np.mean(lats), np.mean(lons)

    def to_dataframe(self) -> pd.DataFrame:
        """
        将轨迹转换为DataFrame

        Returns:
            包含所有点的DataFrame
        """
        data = [point.to_dict() for point in self.points]
        df = pd.DataFrame(data)

        # 添加轨迹ID
        df['trajectory_id'] = self.taxi_id

        return df

    def to_points_array(self) -> np.ndarray:
        """
        将轨迹点转换为numpy数组

        Returns:
            形状为(n_points, 2)的数组
        """
        return np.array([point.to_array() for point in self.points])

    def get_sampling_rate(self) -> float:
        """
        计算轨迹的平均采样率

        Returns:
            采样率（点/小时）
        """
        if self.duration <= 0:
            return 0.0

        return len(self.points) / (self.duration / 3600.0)

    def is_valid(self, min_points: int = 5, min_distance_km: float = 0.1) -> bool:
        """
        检查轨迹是否有效

        Args:
            min_points: 最小点数要求
            min_distance_km: 最小总距离要求（公里）

        Returns:
            如果轨迹有效返回True
        """
        if len(self.points) < min_points:
            return False

        if self.total_distance / 1000.0 < min_distance_km:
            return False

        return True

    def __len__(self) -> int:
        """返回轨迹中的点数"""
        return len(self.points)

    def __repr__(self) -> str:
        """轨迹的字符串表示"""
        return (f"Trajectory(taxi_id={self.taxi_id}, "
                f"points={len(self.points)}, "
                f"distance={self.total_distance / 1000:.2f}km, "
                f"duration={self.duration / 3600:.2f}h)")