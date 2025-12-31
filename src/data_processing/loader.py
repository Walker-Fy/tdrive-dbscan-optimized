"""
T-Drive数据加载器
负责加载和解析原始CSV数据
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Iterator, Optional, Tuple
import os
import csv
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from .trajectory import Trajectory, TrajectoryPoint


class TDriveDataLoader:
    """T-Drive数据集加载器"""

    def __init__(self, data_dir: str, max_workers: int = 4,
                 chunk_size: int = 10000):
        """
        初始化数据加载器

        Args:
            data_dir: 原始数据目录路径
            max_workers: 并行读取的最大工作线程数
            chunk_size: 分块读取的大小
        """
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.trajectory_cache = {}

    def load_trajectories_from_file(self, file_path: Path,
                                    limit: Optional[int] = None) -> List[Trajectory]:
        """
        从单个文件加载轨迹数据

        Args:
            file_path: 数据文件路径
            limit: 限制加载的轨迹数量（用于测试）

        Returns:
            轨迹对象列表
        """
        trajectories = []
        current_taxi_id = None
        current_points = []

        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)

                for i, row in enumerate(reader):
                    if len(row) < 4:
                        continue

                    try:
                        taxi_id = row[0]
                        timestamp = self._parse_timestamp(row[1])
                        lon = float(row[2])
                        lat = float(row[3])

                        # 创建轨迹点
                        point = TrajectoryPoint(
                            latitude=lat,
                            longitude=lon,
                            timestamp=timestamp,
                            taxi_id=taxi_id
                        )

                        # 检查是否开始新的轨迹
                        if current_taxi_id is None:
                            current_taxi_id = taxi_id
                        elif taxi_id != current_taxi_id:
                            # 保存前一个轨迹
                            if current_points:
                                trajectory = Trajectory(
                                    taxi_id=current_taxi_id,
                                    points=current_points.copy()
                                )
                                trajectories.append(trajectory)

                            # 重置当前轨迹
                            current_taxi_id = taxi_id
                            current_points = []

                            if limit and len(trajectories) >= limit:
                                break

                        current_points.append(point)

                    except (ValueError, IndexError) as e:
                        warnings.warn(f"解析第{i}行时出错: {e}")
                        continue

                # 添加最后一个轨迹
                if current_points:
                    trajectory = Trajectory(
                        taxi_id=current_taxi_id,
                        points=current_points
                    )
                    trajectories.append(trajectory)

        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")

        return trajectories

    def load_all_trajectories(self, sample_rate: float = 1.0,
                              limit_per_file: Optional[int] = None) -> List[Trajectory]:
        """
        加载所有轨迹文件

        Args:
            sample_rate: 采样率 (0.0-1.0)
            limit_per_file: 每个文件的最大轨迹数

        Returns:
            所有轨迹的列表
        """
        # 查找所有CSV文件
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            csv_files = list(self.data_dir.glob("*.txt"))

        print(f"找到 {len(csv_files)} 个数据文件")

        all_trajectories = []

        if self.max_workers > 1 and len(csv_files) > 1:
            # 并行加载
            all_trajectories = self._load_parallel(csv_files, sample_rate, limit_per_file)
        else:
            # 串行加载
            for file_path in csv_files[:int(len(csv_files) * sample_rate)]:
                print(f"加载文件: {file_path.name}")
                trajectories = self.load_trajectories_from_file(file_path, limit_per_file)
                all_trajectories.extend(trajectories)

        print(f"总共加载 {len(all_trajectories)} 条轨迹")
        return all_trajectories

    def _load_parallel(self, csv_files: List[Path], sample_rate: float,
                       limit_per_file: Optional[int]) -> List[Trajectory]:
        """
        并行加载轨迹文件

        Args:
            csv_files: CSV文件列表
            sample_rate: 采样率
            limit_per_file: 每个文件的最大轨迹数

        Returns:
            合并后的轨迹列表
        """
        # 采样文件
        sampled_files = csv_files[:int(len(csv_files) * sample_rate)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file_path in sampled_files:
                future = executor.submit(
                    self.load_trajectories_from_file,
                    file_path,
                    limit_per_file
                )
                futures.append(future)

            all_trajectories = []
            for i, future in enumerate(futures):
                try:
                    trajectories = future.result()
                    all_trajectories.extend(trajectories)
                    print(f"完成文件 {i + 1}/{len(sampled_files)}: 加载了 {len(trajectories)} 条轨迹")
                except Exception as e:
                    print(f"处理文件 {sampled_files[i].name} 时出错: {e}")

        return all_trajectories

    def load_as_dataframe(self, file_path: Path,
                          nrows: Optional[int] = None) -> pd.DataFrame:
        """
        将数据加载为Pandas DataFrame

        Args:
            file_path: 数据文件路径
            nrows: 限制加载的行数

        Returns:
            包含轨迹数据的DataFrame
        """
        try:
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                header=None,
                names=['taxi_id', 'timestamp', 'lon', 'lat'],
                dtype={'taxi_id': str, 'timestamp': str, 'lon': float, 'lat': float}
            )
            return df
        except Exception as e:
            print(f"加载DataFrame时出错: {e}")
            return pd.DataFrame()

    def chunked_loader(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """
        分块加载大数据文件

        Args:
            file_path: 数据文件路径

        Yields:
            数据块（DataFrame）
        """
        chunk_iterator = pd.read_csv(
            file_path,
            chunksize=self.chunk_size,
            header=None,
            names=['taxi_id', 'timestamp', 'lon', 'lat'],
            dtype={'taxi_id': str, 'timestamp': str, 'lon': float, 'lat': float}
        )

        for chunk in chunk_iterator:
            yield chunk

    def _parse_timestamp(self, timestamp_str: str) -> pd.Timestamp:
        """
        解析时间戳字符串

        Args:
            timestamp_str: 时间戳字符串（格式: YYYY-MM-DD HH:MM:SS）

        Returns:
            pandas Timestamp对象
        """
        try:
            # 移除可能的空格和换行符
            timestamp_str = timestamp_str.strip()

            # 尝试不同的格式
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M'
            ]

            for fmt in formats:
                try:
                    return pd.to_datetime(timestamp_str, format=fmt)
                except:
                    continue

            # 如果所有格式都失败，尝试自动解析
            return pd.to_datetime(timestamp_str)

        except Exception as e:
            warnings.warn(f"解析时间戳 '{timestamp_str}' 失败: {e}")
            return pd.NaT

    def get_dataset_info(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            包含数据集信息的字典
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            csv_files = list(self.data_dir.glob("*.txt"))

        info = {
            'n_files': len(csv_files),
            'file_names': [f.name for f in csv_files],
            'total_size_mb': sum(f.stat().st_size for f in csv_files) / (1024 * 1024),
            'data_dir': str(self.data_dir)
        }

        return info


def load_tdrive_csv(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    快速加载T-Drive CSV文件的便捷函数

    Args:
        file_path: CSV文件路径
        nrows: 限制加载的行数

    Returns:
        包含轨迹数据的DataFrame
    """
    loader = TDriveDataLoader(os.path.dirname(file_path))
    return loader.load_as_dataframe(Path(file_path), nrows)


def save_preprocessed_data(trajectories: List[Trajectory],
                           output_path: str,
                           format: str = 'parquet') -> None:
    """
    保存预处理后的轨迹数据

    Args:
        trajectories: 轨迹列表
        output_path: 输出文件路径
        format: 输出格式 ('parquet', 'csv', 'pickle')
    """
    # 转换为DataFrame
    data = []
    for traj in trajectories:
        for point in traj.points:
            data.append({
                'taxi_id': traj.taxi_id,
                'timestamp': point.timestamp,
                'latitude': point.latitude,
                'longitude': point.longitude,
                'speed': point.speed if hasattr(point, 'speed') else None,
                'direction': point.direction if hasattr(point, 'direction') else None
            })

    df = pd.DataFrame(data)

    # 保存为指定格式
    output_path = Path(output_path)

    if format == 'parquet':
        df.to_parquet(output_path.with_suffix('.parquet'))
    elif format == 'csv':
        df.to_csv(output_path.with_suffix('.csv'), index=False)
    elif format == 'pickle':
        df.to_pickle(output_path.with_suffix('.pkl'))
    else:
        raise ValueError(f"不支持的格式: {format}")

    print(f"数据已保存到: {output_path}")