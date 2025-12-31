"""
并行工作进程管理
DBSCAN算法的并行任务执行
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from typing import List, Tuple, Dict, Optional, Any, Callable
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum
import warnings

from ..clustering.dbscan_sequential import DBSCANSequential
from .partitioning import Partition


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """计算任务"""
    id: int
    partition: Partition
    eps: float
    min_samples: int
    metric: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """任务持续时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class DBSCANWorker:
    """DBSCAN工作进程"""

    def __init__(self, worker_id: int):
        """
        初始化工作进程

        Args:
            worker_id: 工作进程ID
        """
        self.worker_id = worker_id
        self.n_tasks_completed = 0
        self.total_processing_time = 0.0

    def process_partition(self, task: Task) -> Dict[str, Any]:
        """
        处理一个数据分区

        Args:
            task: 计算任务

        Returns:
            处理结果字典
        """
        try:
            start_time = time.time()

            # 创建DBSCAN聚类器
            dbscan = DBSCANSequential(
                eps=task.eps,
                min_samples=task.min_samples,
                metric=task.metric
            )

            # 在分区数据上执行聚类
            dbscan.fit(task.partition.points)

            # 获取聚类结果
            stats = dbscan.get_cluster_stats()

            # 创建结果字典
            result = {
                'partition_id': task.partition.id,
                'worker_id': self.worker_id,
                'labels': dbscan.labels_,
                'core_sample_indices': dbscan.core_sample_indices_,
                'n_points': len(task.partition.points),
                'n_clusters': stats['n_clusters'],
                'n_noise': stats['n_noise'],
                'processing_time': time.time() - start_time,
                'success': True
            }

            # 更新工作进程统计
            self.n_tasks_completed += 1
            self.total_processing_time += result['processing_time']

            return result

        except Exception as e:
            error_msg = f"Worker {self.worker_id} 处理分区 {task.partition.id} 时出错: {str(e)}"
            warnings.warn(error_msg)

            return {
                'partition_id': task.partition.id,
                'worker_id': self.worker_id,
                'error': error_msg,
                'success': False
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取工作进程统计信息

        Returns:
            统计信息字典
        """
        return {
            'worker_id': self.worker_id,
            'n_tasks_completed': self.n_tasks_completed,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': (self.total_processing_time / self.n_tasks_completed
                                    if self.n_tasks_completed > 0 else 0)
        }


class WorkerPool:
    """工作进程池管理器"""

    def __init__(self, n_workers: int = -1,
                 worker_class: Callable = DBSCANWorker):
        """
        初始化工作进程池

        Args:
            n_workers: 工作进程数量，-1表示使用所有CPU核心
            worker_class: 工作进程类
        """
        self.n_workers = mp.cpu_count() if n_workers == -1 else n_workers
        self.worker_class = worker_class
        self.workers: List[DBSCANWorker] = []
        self.task_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self.processes: List[mp.Process] = []

        # 性能统计
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def initialize(self) -> None:
        """初始化工作进程池"""
        # 创建工作进程
        self.workers = [self.worker_class(i) for i in range(self.n_workers)]

        # 创建任务队列和结果队列
        manager = Manager()
        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()

    def _worker_process(self, worker_id: int, task_queue: Queue,
                        result_queue: Queue) -> None:
        """
        工作进程函数（运行在子进程中）

        Args:
            worker_id: 工作进程ID
            task_queue: 任务队列
            result_queue: 结果队列
        """
        worker = self.worker_class(worker_id)

        while True:
            try:
                # 从队列获取任务
                task = task_queue.get(timeout=1.0)

                if task is None:  # 终止信号
                    break

                # 更新任务状态
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()

                # 处理任务
                result = worker.process_partition(task)

                # 更新任务状态
                task.status = TaskStatus.COMPLETED
                task.end_time = time.time()
                task.result = result

                # 发送结果
                result_queue.put(task)

            except Exception as e:
                if isinstance(e, mp.queues.Empty):
                    continue  # 队列为空，继续等待

                # 处理错误
                if 'task' in locals():
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.end_time = time.time()
                    result_queue.put(task)

                warnings.warn(f"Worker {worker_id} 出错: {e}")

    def start(self) -> None:
        """启动工作进程池"""
        if not self.workers:
            self.initialize()

        self.start_time = time.time()

        # 创建并启动进程
        for worker_id in range(self.n_workers):
            process = mp.Process(
                target=self._worker_process,
                args=(worker_id, self.task_queue, self.result_queue),
                daemon=True
            )
            process.start()
            self.processes.append(process)

        print(f"启动了 {self.n_workers} 个工作进程")

    def submit_tasks(self, tasks: List[Task]) -> None:
        """
        提交任务到工作进程池

        Args:
            tasks: 任务列表
        """
        if not self.task_queue:
            raise RuntimeError("工作进程池未初始化")

        for task in tasks:
            self.task_queue.put(task)

        print(f"提交了 {len(tasks)} 个任务")

    def get_results(self, timeout: Optional[float] = None) -> List[Task]:
        """
        获取所有任务结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            完成任务列表
        """
        if not self.result_queue:
            raise RuntimeError("工作进程池未初始化")

        completed_tasks = []
        start_time = time.time()

        while True:
            try:
                # 设置超时
                if timeout:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        break

                    task = self.result_queue.get(timeout=min(1.0, remaining))
                else:
                    task = self.result_queue.get(timeout=1.0)

                completed_tasks.append(task)

            except mp.queues.Empty:
                # 检查是否所有任务都已完成
                if self.task_queue.empty():
                    # 给进程一些时间完成剩余工作
                    time.sleep(0.1)
                    if self.result_queue.empty():
                        break

        return completed_tasks

    def stop(self) -> None:
        """停止工作进程池"""
        self.end_time = time.time()

        # 发送终止信号
        if self.task_queue:
            for _ in range(self.n_workers):
                self.task_queue.put(None)

        # 等待进程结束
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()

        self.processes.clear()
        print("工作进程池已停止")

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        获取进程池统计信息

        Returns:
            统计信息字典
        """
        total_tasks = 0
        total_processing_time = 0.0
        successful_tasks = 0
        failed_tasks = 0

        for worker in self.workers:
            stats = worker.get_stats()
            total_tasks += stats['n_tasks_completed']
            total_processing_time += stats['total_processing_time']

        return {
            'n_workers': self.n_workers,
            'total_tasks_completed': total_tasks,
            'total_processing_time': total_processing_time,
            'avg_processing_time_per_task': (total_processing_time / total_tasks
                                             if total_tasks > 0 else 0),
            'pool_duration': (self.end_time - self.start_time
                              if self.start_time and self.end_time else 0)
        }


class TaskScheduler:
    """任务调度器（实现动态负载均衡）"""

    def __init__(self, worker_pool: WorkerPool,
                 load_balance_threshold: float = 0.3):
        """
        初始化任务调度器

        Args:
            worker_pool: 工作进程池
            load_balance_threshold: 负载均衡阈值
        """
        self.worker_pool = worker_pool
        self.load_balance_threshold = load_balance_threshold
        self.task_history: Dict[int, List[float]] = {}

    def schedule_tasks(self, tasks: List[Task],
                       strategy: str = 'round_robin') -> None:
        """
        调度任务到工作进程

        Args:
            tasks: 任务列表
            strategy: 调度策略 ('round_robin', 'size_based', 'dynamic')
        """
        if strategy == 'round_robin':
            self._round_robin_schedule(tasks)
        elif strategy == 'size_based':
            self._size_based_schedule(tasks)
        elif strategy == 'dynamic':
            self._dynamic_schedule(tasks)
        else:
            raise ValueError(f"未知的调度策略: {strategy}")

    def _round_robin_schedule(self, tasks: List[Task]) -> None:
        """
        轮询调度

        Args:
            tasks: 任务列表
        """
        # 直接提交所有任务，由工作进程池自行调度
        self.worker_pool.submit_tasks(tasks)

    def _size_based_schedule(self, tasks: List[Task]) -> None:
        """
        基于任务大小的调度

        Args:
            tasks: 任务列表
        """
        # 按分区大小排序（从大到小）
        sorted_tasks = sorted(tasks, key=lambda t: t.partition.n_points, reverse=True)

        # 将大任务分配给空闲的工作进程
        self.worker_pool.submit_tasks(sorted_tasks)

    def _dynamic_schedule(self, tasks: List[Task]) -> None:
        """
        动态负载均衡调度

        Args:
            tasks: 任务列表
        """
        # 根据历史性能预测任务执行时间
        weighted_tasks = []

        for task in tasks:
            # 根据分区大小估计处理时间
            estimated_time = self._estimate_processing_time(task)
            weighted_tasks.append((task, estimated_time))

        # 按估计时间排序
        weighted_tasks.sort(key=lambda x: x[1], reverse=True)

        # 提交任务
        self.worker_pool.submit_tasks([task for task, _ in weighted_tasks])

    def _estimate_processing_time(self, task: Task) -> float:
        """
        估计任务处理时间

        Args:
            task: 计算任务

        Returns:
            估计的处理时间
        """
        # 基于分区大小的简单估计
        base_time_per_point = 0.001  # 每个点的基准处理时间（秒）
        estimated_time = task.partition.n_points * base_time_per_point

        # 考虑距离度量复杂度
        if task.metric == 'haversine':
            estimated_time *= 2.0  # Haversine距离计算更耗时

        return estimated_time

    def monitor_load(self) -> Dict[int, float]:
        """
        监控工作进程负载

        Returns:
            工作进程ID到负载的映射
        """
        load_info = {}

        for worker in self.worker_pool.workers:
            stats = worker.get_stats()
            # 使用总处理时间作为负载指标
            load_info[worker.worker_id] = stats['total_processing_time']

        return load_info

    def rebalance_if_needed(self) -> bool:
        """
        检查并执行负载重平衡

        Returns:
            是否执行了重平衡
        """
        load_info = self.monitor_load()

        if not load_info:
            return False

        # 计算负载方差
        loads = list(load_info.values())
        avg_load = np.mean(loads)

        if avg_load == 0:
            return False

        # 计算负载不均衡度
        load_ratios = [load / avg_load for load in loads]
        imbalance = max(load_ratios) - min(load_ratios)

        if imbalance > self.load_balance_threshold:
            print(f"检测到负载不均衡: {imbalance:.3f}，执行重平衡...")
            # 这里可以实现更复杂的重平衡逻辑
            return True

        return False


def merge_partition_results(completed_tasks: List[Task],
                            eps: float) -> Dict[str, Any]:
    """
    合并分区聚类结果

    Args:
        completed_tasks: 完成的任务列表
        eps: DBSCAN邻域半径

    Returns:
        合并后的聚类结果
    """
    if not completed_tasks:
        return {}

    # 收集所有成功的结果
    successful_tasks = [t for t in completed_tasks if t.result and t.result.get('success', False)]

    if not successful_tasks:
        return {'success': False, 'error': '所有任务都失败了'}

    # 初始化全局标签
    total_points = 0
    for task in successful_tasks:
        total_points += task.result['n_points']

    global_labels = np.full(total_points, -1, dtype=np.int32)
    point_offset = 0

    # 合并标签（简单的偏移合并）
    for task in successful_tasks:
        partition_size = task.partition.n_points
        partition_labels = task.result['labels']

        # 应用偏移
        if len(partition_labels) == partition_size:
            global_labels[point_offset:point_offset + partition_size] = partition_labels
            point_offset += partition_size

    # 重新映射标签以避免冲突
    unique_labels = np.unique(global_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}

    for i in range(len(global_labels)):
        global_labels[i] = label_mapping[global_labels[i]]

    # 计算统计信息
    n_clusters = len([l for l in unique_labels if l != 0])
    n_noise = np.sum(global_labels == 0)

    return {
        'success': True,
        'global_labels': global_labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_points': total_points,
        'n_partitions': len(successful_tasks),
        'n_failed_partitions': len(completed_tasks) - len(successful_tasks)
    }