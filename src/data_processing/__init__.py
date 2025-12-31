"""
数据处理模块
T-Drive轨迹数据的加载、预处理和特征提取
"""

from .loader import TDriveDataLoader, load_tdrive_csv, save_preprocessed_data
from .preprocessor import TrajectoryPreprocessor, clean_trajectory_data, resample_trajectories
from .trajectory import Trajectory, TrajectoryPoint, TrajectorySegment

__all__ = [
    'TDriveDataLoader',
    'load_tdrive_csv',
    'save_preprocessed_data',
    'TrajectoryPreprocessor',
    'clean_trajectory_data',
    'resample_trajectories',
    'Trajectory',
    'TrajectoryPoint',
    'TrajectorySegment'
]