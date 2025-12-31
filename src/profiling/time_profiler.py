"""
时间性能分析器
分析DBSCAN算法的执行时间和性能瓶颈
"""

import time
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import wraps
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class TimeMeasurement:
    """时间测量结果"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    children: List['TimeMeasurement'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def stop(self) -> float:
        """停止计时并返回持续时间"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration


class TimeProfiler:
    """时间性能分析器"""

    def __init__(self, enable_profiling: bool = True):
        """
        初始化时间分析器

        Args:
            enable_profiling: 是否启用详细性能分析
        """
        self.enable_profiling = enable_profiling
        self.measurements: List[TimeMeasurement] = []
        self.current_stack: List[TimeMeasurement] = []
        self.function_timings: Dict[str, List[float]] = defaultdict(list)
        self.profiler: Optional[cProfile.Profile] = None

        # 性能统计
        self.total_execution_time = 0.0
        self.n_calls = 0

    def start(self, name: str) -> TimeMeasurement:
        """
        开始计时

        Args:
            name: 测量名称

        Returns:
            时间测量对象
        """
        measurement = TimeMeasurement(
            name=name,
            start_time=time.time()
        )

        # 添加到当前堆栈
        if self.current_stack:
            parent = self.current_stack[-1]
            parent.children.append(measurement)
        else:
            self.measurements.append(measurement)

        self.current_stack.append(measurement)

        return measurement

    def stop(self, name: str = None) -> Optional[float]:
        """
        停止计时

        Args:
            name: 要停止的测量名称（如果为None则停止当前）

        Returns:
            持续时间（秒）
        """
        if not self.current_stack:
            return None

        if name is None:
            # 停止当前测量
            measurement = self.current_stack.pop()
        else:
            # 查找并停止指定名称的测量
            for i, meas in enumerate(reversed(self.current_stack)):
                if meas.name == name:
                    measurement = self.current_stack.pop(-i - 1)

                    # 弹出中间的所有测量
                    for _ in range(i):
                        self.current_stack.pop().stop()
                    break
            else:
                warnings.warn(f"未找到测量 '{name}'")
                return None

        duration = measurement.stop()

        # 记录函数计时
        self.function_timings[measurement.name].append(duration)
        self.total_execution_time += duration
        self.n_calls += 1

        return duration

    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        分析函数的执行时间

        Args:
            func: 要分析的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            (函数结果, 性能分析结果)
        """
        # 开始分析
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # 开始计时
        measurement = self.start(f"{func.__name__}")

        try:
            # 执行函数
            result = func(*args, **kwargs)
        finally:
            # 停止计时
            self.stop(measurement.name)

            # 停止分析
            if self.enable_profiling and self.profiler:
                self.profiler.disable()

        # 生成分析结果
        analysis = self._analyze_function_performance(func.__name__)

        return result, analysis

    def _analyze_function_performance(self, function_name: str) -> Dict[str, Any]:
        """
        分析函数性能

        Args:
            function_name: 函数名

        Returns:
            性能分析结果
        """
        analysis = {
            'function_name': function_name,
            'execution_time': self.function_timings[function_name][-1] if function_name in self.function_timings else 0,
            'n_calls': len(self.function_timings.get(function_name, [])),
            'avg_time': np.mean(self.function_timings.get(function_name, [0])),
            'std_time': np.std(self.function_timings.get(function_name, [0])),
            'min_time': np.min(self.function_timings.get(function_name, [0])),
            'max_time': np.max(self.function_timings.get(function_name, [0]))
        }

        # 如果有cProfile数据，添加详细信息
        if self.profiler:
            profile_data = self._get_profile_stats()
            analysis.update({
                'profile_data': profile_data,
                'top_functions': profile_data.get('top_functions', [])[:5]
            })

        return analysis

    def _get_profile_stats(self) -> Dict[str, Any]:
        """
        获取cProfile统计信息

        Returns:
            profile统计信息
        """
        if not self.profiler:
            return {}

        # 获取profile统计
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # 显示前20个函数

        # 解析统计信息
        stats_output = s.getvalue()

        # 提取主要信息
        lines = stats_output.split('\n')
        top_functions = []

        for line in lines[5:25]:  # 跳过标题行
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        ncalls = parts[0]
                        tottime = float(parts[2])
                        percall = float(parts[3])
                        cumtime = float(parts[4])
                        function_name = ' '.join(parts[5:])

                        top_functions.append({
                            'function': function_name,
                            'ncalls': ncalls,
                            'tottime': tottime,
                            'percall': percall,
                            'cumtime': cumtime
                        })
                    except (ValueError, IndexError):
                        continue

        return {
            'stats_output': stats_output,
            'top_functions': top_functions
        }

    def analyze_performance_bottlenecks(self, threshold_ratio: float = 0.1) -> List[Dict[str, Any]]:
        """
        分析性能瓶颈

        Args:
            threshold_ratio: 阈值比例（超过总时间此比例的函数被视为瓶颈）

        Returns:
            瓶颈函数列表
        """
        bottlenecks = []

        if not self.function_timings or self.total_execution_time == 0:
            return bottlenecks

        # 计算每个函数的总时间
        function_total_times = {}
        for func_name, timings in self.function_timings.items():
            function_total_times[func_name] = np.sum(timings)

        # 识别瓶颈
        for func_name, total_time in sorted(function_total_times.items(),
                                            key=lambda x: x[1], reverse=True):
            time_ratio = total_time / self.total_execution_time

            if time_ratio > threshold_ratio:
                bottlenecks.append({
                    'function': func_name,
                    'total_time': total_time,
                    'time_ratio': time_ratio,
                    'n_calls': len(self.function_timings[func_name]),
                    'avg_time_per_call': total_time / len(self.function_timings[func_name])
                })

        return bottlenecks

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        生成性能分析报告

        Returns:
            性能报告
        """
        # 分析瓶颈
        bottlenecks = self.analyze_performance_bottlenecks()

        # 生成调用树
        call_tree = self._generate_call_tree()

        # 计算总体统计
        total_stats = {
            'total_execution_time': self.total_execution_time,
            'n_function_calls': self.n_calls,
            'n_unique_functions': len(self.function_timings),
            'n_bottlenecks': len(bottlenecks)
        }

        # 最耗时的函数
        top_functions = []
        for func_name, timings in self.function_timings.items():
            total_time = np.sum(timings)
            top_functions.append({
                'function': func_name,
                'total_time': total_time,
                'n_calls': len(timings),
                'avg_time': np.mean(timings)
            })

        top_functions.sort(key=lambda x: x['total_time'], reverse=True)

        report = {
            'summary': total_stats,
            'top_functions': top_functions[:10],
            'bottlenecks': bottlenecks,
            'call_tree': call_tree,
            'function_timings': {
                func: {
                    'total': np.sum(times),
                    'avg': np.mean(times),
                    'std': np.std(times),
                    'n_calls': len(times)
                }
                for func, times in self.function_timings.items()
            }
        }

        return report

    def _generate_call_tree(self) -> List[Dict[str, Any]]:
        """
        生成函数调用树

        Returns:
            调用树结构
        """

        def process_measurement(meas: TimeMeasurement, depth: int = 0) -> Dict[str, Any]:
            """递归处理测量"""
            node = {
                'name': meas.name,
                'duration': meas.duration or 0,
                'depth': depth,
                'children': []
            }

            for child in meas.children:
                child_node = process_measurement(child, depth + 1)
                node['children'].append(child_node)

            return node

        # 从根测量开始
        call_tree = []
        for meas in self.measurements:
            node = process_measurement(meas)
            call_tree.append(node)

        return call_tree

    def suggest_optimizations(self) -> List[str]:
        """
        提供性能优化建议

        Returns:
            优化建议列表
        """
        suggestions = []

        # 分析瓶颈
        bottlenecks = self.analyze_performance_bottlenecks()

        for bottleneck in bottlenecks:
            func_name = bottleneck['function']
            time_ratio = bottleneck['time_ratio']

            if time_ratio > 0.3:
                suggestions.append(
                    f"函数 '{func_name}' 占用 {time_ratio:.1%} 的执行时间，是主要瓶颈"
                )

            if bottleneck['n_calls'] > 1000:
                suggestions.append(
                    f"函数 '{func_name}' 被调用 {bottleneck['n_calls']} 次，考虑减少调用次数或使用向量化"
                )

        # 根据函数名提供具体建议
        for func_name in self.function_timings:
            if 'distance' in func_name.lower():
                suggestions.append(
                    f"距离计算函数 '{func_name}' 可能受益于KDTree或BallTree索引"
                )

            if 'neighbor' in func_name.lower():
                suggestions.append(
                    f"邻居搜索函数 '{func_name}' 可以考虑使用空间索引优化"
                )

            if 'loop' in func_name.lower() or 'for' in func_name:
                suggestions.append(
                    f"循环函数 '{func_name}' 可以考虑向量化或并行化"
                )

        # 通用优化建议
        suggestions.extend([
            "使用numpy向量化操作代替Python循环",
            "考虑使用JIT编译（如Numba）加速数值计算",
            "使用适当的数据结构（如KDTree进行空间搜索）",
            "并行化独立计算任务",
            "缓存重复计算结果"
        ])

        return list(set(suggestions))  # 去重

    def reset(self) -> None:
        """重置分析器"""
        self.measurements.clear()
        self.current_stack.clear()
        self.function_timings.clear()
        self.total_execution_time = 0.0
        self.n_calls = 0
        self.profiler = None


def profile_function(func: Callable = None, detailed: bool = True):
    """
    装饰器：分析函数执行时间

    Args:
        func: 要装饰的函数
        detailed: 是否进行详细分析
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            profiler = TimeProfiler(enable_profiling=detailed)

            # 分析函数性能
            result, analysis = profiler.profile_function(f, *args, **kwargs)

            # 打印分析结果
            if detailed:
                print(f"\n函数 {f.__name__} 性能分析:")
                print(f"  执行时间: {analysis['execution_time']:.4f} 秒")

                if 'top_functions' in analysis:
                    print(f"  主要耗时函数:")
                    for func_info in analysis['top_functions'][:3]:
                        print(f"    {func_info['function']}: {func_info['cumtime']:.4f} 秒")

            # 返回结果和性能数据
            return result, analysis

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def analyze_performance_bottlenecks(implementation_func: Callable,
                                    test_cases: List[Dict]) -> Dict[str, Any]:
    """
    分析实现的性能瓶颈

    Args:
        implementation_func: 实现函数
        test_cases: 测试用例列表

    Returns:
        瓶颈分析结果
    """
    profiler = TimeProfiler(enable_profiling=True)
    results = []

    for i, test_case in enumerate(test_cases):
        print(f"运行测试用例 {i + 1}/{len(test_cases)}...")

        # 分析函数性能
        result, analysis = profiler.profile_function(
            implementation_func,
            **test_case
        )

        results.append({
            'test_case': test_case.get('name', f'test_{i}'),
            'analysis': analysis,
            'result': result if isinstance(result, (int, float, str)) else str(type(result))
        })

    # 综合分析
    bottlenecks = profiler.analyze_performance_bottlenecks()
    suggestions = profiler.suggest_optimizations()

    return {
        'test_results': results,
        'overall_bottlenecks': bottlenecks,
        'optimization_suggestions': suggestions,
        'summary': {
            'n_tests': len(test_cases),
            'total_execution_time': profiler.total_execution_time,
            'avg_execution_time': profiler.total_execution_time / len(test_cases)
            if test_cases else 0
        }
    }