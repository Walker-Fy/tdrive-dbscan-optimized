"""
聚类结果可视化
轨迹聚类结果的空间和统计可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
from pathlib import Path
import warnings
from scipy.spatial import ConvexHull

try:
    import folium
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    warnings.warn("地理可视化库未安装，部分功能将不可用")


class ClusterVisualizer:
    """聚类可视化器"""

    def __init__(self, figsize: Tuple[int, int] = (12, 10),
                 colormap: str = 'tab20'):
        """
        初始化可视化器

        Args:
            figsize: 图形大小
            colormap: 颜色映射
        """
        self.figsize = figsize
        self.colormap = colormap
        self.cmap = cm.get_cmap(colormap)

    def plot_clusters_2d(self, points: np.ndarray, labels: np.ndarray,
                         title: str = "DBSCAN聚类结果",
                         save_path: Optional[str] = None,
                         show_noise: bool = True,
                         alpha: float = 0.6,
                         s: float = 10.0) -> plt.Figure:
        """
        绘制2D聚类结果

        Args:
            points: 点数据，形状为(n, 2)
            labels: 聚类标签，形状为(n,)
            title: 图表标题
            save_path: 保存路径
            show_noise: 是否显示噪声点
            alpha: 透明度
            s: 点的大小

        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 获取唯一的标签（排除噪声点标签0）
        unique_labels = np.unique(labels)

        # 为每个聚类分配颜色
        colors = self.cmap(np.linspace(0, 1, len(unique_labels) if 0 not in unique_labels else len(unique_labels) - 1))

        # 绘制每个聚类
        for i, label in enumerate(unique_labels):
            if label == 0 and not show_noise:
                continue

            # 获取属于当前聚类的点
            cluster_mask = labels == label

            if np.sum(cluster_mask) == 0:
                continue

            cluster_points = points[cluster_mask]

            # 选择颜色
            if label == 0:
                color = 'gray'
                label_text = '噪声点'
                marker = 'x'
                size = s * 0.5
                alpha_cluster = alpha * 0.5
            else:
                color = colors[i - 1 if 0 in unique_labels else i]
                label_text = f'聚类 {label}'
                marker = 'o'
                size = s
                alpha_cluster = alpha

            # 绘制点
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=[color], label=label_text,
                       marker=marker, s=size, alpha=alpha_cluster, edgecolors='w', linewidths=0.5)

            # 绘制凸包（对于较大的聚类）
            if label != 0 and len(cluster_points) > 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # 闭合多边形

                    ax.plot(hull_points[:, 0], hull_points[:, 1],
                            color=color, alpha=0.3, linewidth=1, linestyle='--')
                except:
                    pass

        # 设置图表属性
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('经度' if points.shape[1] == 2 else 'X坐标')
        ax.set_ylabel('纬度' if points.shape[1] == 2 else 'Y坐标')
        ax.grid(True, alpha=0.3)

        # 添加图例（只显示前10个聚类以避免过于拥挤）
        handles, labels_legend = ax.get_legend_handles_labels()
        if len(handles) > 15:
            ax.legend(handles[:15], labels_legend[:15], loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right')

        # 添加统计信息
        n_clusters = len([l for l in unique_labels if l != 0])
        n_noise = np.sum(labels == 0)
        n_points = len(points)

        stats_text = f'聚类数: {n_clusters}\n噪声点: {n_noise}\n总点数: {n_points}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚类图已保存到: {save_path}")

        return fig

    def plot_cluster_heatmap(self, points: np.ndarray, labels: np.ndarray,
                             title: str = "聚类密度热力图",
                             save_path: Optional[str] = None,
                             bins: int = 50,
                             cmap: str = 'YlOrRd') -> plt.Figure:
        """
        绘制聚类密度热力图

        Args:
            points: 点数据
            labels: 聚类标签
            title: 图表标题
            save_path: 保存路径
            bins: 直方图分箱数
            cmap: 颜色映射

        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 整体密度热力图
        ax1 = axes[0, 0]
        hb1 = ax1.hexbin(points[:, 0], points[:, 1], gridsize=bins, cmap=cmap, alpha=0.8)
        ax1.set_title('整体点密度热力图', fontsize=12)
        ax1.set_xlabel('经度')
        ax1.set_ylabel('纬度')
        cb1 = fig.colorbar(hb1, ax=ax1)
        cb1.set_label('点数')

        # 2. 非噪声点密度热力图
        ax2 = axes[0, 1]
        non_noise_mask = labels != 0
        if np.any(non_noise_mask):
            non_noise_points = points[non_noise_mask]
            hb2 = ax2.hexbin(non_noise_points[:, 0], non_noise_points[:, 1],
                             gridsize=bins, cmap=cmap, alpha=0.8)
            ax2.set_title('聚类点密度热力图', fontsize=12)
            ax2.set_xlabel('经度')
            ax2.set_ylabel('纬度')
            cb2 = fig.colorbar(hb2, ax=ax2)
            cb2.set_label('点数')

        # 3. 聚类大小分布
        ax3 = axes[1, 0]
        unique_labels = np.unique(labels[labels != 0])
        cluster_sizes = []
        cluster_ids = []

        for label in unique_labels:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
            cluster_ids.append(label)

        if cluster_sizes:
            bars = ax3.bar(range(len(cluster_sizes)), cluster_sizes,
                           color=self.cmap(np.linspace(0, 1, len(cluster_sizes))))
            ax3.set_title('聚类大小分布', fontsize=12)
            ax3.set_xlabel('聚类ID')
            ax3.set_ylabel('点数')
            ax3.set_xticks(range(len(cluster_ids)))
            ax3.set_xticklabels(cluster_ids, rotation=45)

            # 在柱状图上添加数值
            for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{size}', ha='center', va='bottom', fontsize=8)

        # 4. 聚类质心分布
        ax4 = axes[1, 1]
        if cluster_sizes:
            centroids = []
            for label in unique_labels:
                cluster_points = points[labels == label]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)

            centroids = np.array(centroids)
            scatter = ax4.scatter(centroids[:, 0], centroids[:, 1],
                                  c=cluster_sizes, s=np.sqrt(cluster_sizes),
                                  cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
            ax4.set_title('聚类质心分布', fontsize=12)
            ax4.set_xlabel('经度')
            ax4.set_ylabel('纬度')
            cb4 = fig.colorbar(scatter, ax=ax4)
            cb4.set_label('聚类大小')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {save_path}")

        return fig

    def plot_interactive_map(self, points: np.ndarray, labels: np.ndarray,
                             save_path: Optional[str] = None,
                             center: Optional[Tuple[float, float]] = None,
                             zoom_start: int = 12):
        """
        创建交互式聚类地图（需要folium）

        Args:
            points: 点数据 [纬度, 经度]
            labels: 聚类标签
            save_path: 保存路径
            center: 地图中心点 [纬度, 经度]
            zoom_start: 初始缩放级别

        Returns:
            folium地图对象
        """
        if not GEO_AVAILABLE:
            warnings.warn("folium未安装，无法创建交互式地图")
            return None

        # 确定地图中心
        if center is None:
            center = [np.mean(points[:, 0]), np.mean(points[:, 1])]

        # 创建地图
        m = folium.Map(location=center, zoom_start=zoom_start,
                       tiles='cartodbpositron')

        # 获取唯一的聚类标签
        unique_labels = np.unique(labels)

        # 为每个聚类创建颜色映射
        colors = {}
        for i, label in enumerate(unique_labels):
            if label == 0:
                colors[label] = '#808080'  # 灰色表示噪声点
            else:
                # 从colormap获取颜色并转换为十六进制
                color = self.cmap((i - 1) / (len(unique_labels) - 1) if 0 in unique_labels else i / len(unique_labels))
                colors[label] = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                )

        # 添加聚类点
        for i in range(len(points)):
            lat, lon = points[i, 0], points[i, 1]
            label = labels[i]

            # 创建弹出窗口内容
            popup_text = f"聚类: {label}<br>纬度: {lat:.6f}<br>经度: {lon:.6f}"

            # 根据聚类标签设置颜色和大小
            if label == 0:
                color = colors[label]
                radius = 2
                fill_opacity = 0.3
            else:
                color = colors[label]
                radius = 4
                fill_opacity = 0.6

            # 添加圆形标记
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                weight=1
            ).add_to(m)

        # 添加聚类图例
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: auto;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;">
            <b>聚类图例</b><br>
        '''

        for label in unique_labels[:10]:  # 只显示前10个聚类
            if label == 0:
                legend_html += f'<i style="background:{colors[label]}; width:20px; height:20px; display:inline-block;"></i> 噪声点<br>'
            else:
                legend_html += f'<i style="background:{colors[label]}; width:20px; height:20px; display:inline-block;"></i> 聚类 {label}<br>'

        if len(unique_labels) > 10:
            legend_html += f'<i>... 还有 {len(unique_labels) - 10} 个聚类</i><br>'

        legend_html += f'<br><b>统计信息</b><br>'
        legend_html += f'总点数: {len(points)}<br>'
        legend_html += f'聚类数: {len([l for l in unique_labels if l != 0])}<br>'
        legend_html += f'噪声点: {np.sum(labels == 0)}'
        legend_html += '</div>'

        m.get_root().html.add_child(folium.Element(legend_html))

        # 保存地图
        if save_path:
            m.save(save_path)
            print(f"交互式地图已保存到: {save_path}")

        return m

    def plot_cluster_statistics(self, points: np.ndarray, labels: np.ndarray,
                                title: str = "聚类统计分析",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制聚类统计信息

        Args:
            points: 点数据
            labels: 聚类标签
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib图形对象
        """
        # 计算聚类统计信息
        unique_labels = np.unique(labels)
        cluster_stats = []

        for label in unique_labels:
            if label == 0:
                continue

            cluster_mask = labels == label
            cluster_points = points[cluster_mask]

            if len(cluster_points) < 2:
                continue

            # 计算统计信息
            stats = {
                'label': label,
                'size': len(cluster_points),
                'density': len(cluster_points) / (np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0]) + 1e-10) /
                           (np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1]) + 1e-10),
                'radius': np.max(np.linalg.norm(cluster_points - np.mean(cluster_points, axis=0), axis=1)),
                'compactness': len(cluster_points) / (
                            np.max(np.linalg.norm(cluster_points - np.mean(cluster_points, axis=0), axis=1)) + 1e-10)
            }

            # 计算凸包面积
            try:
                hull = ConvexHull(cluster_points)
                stats['convex_hull_area'] = hull.volume
            except:
                stats['convex_hull_area'] = 0

            cluster_stats.append(stats)

        if not cluster_stats:
            print("没有聚类数据可分析")
            return plt.figure()

        # 创建DataFrame
        df_stats = pd.DataFrame(cluster_stats)

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. 聚类大小分布
        ax1 = axes[0, 0]
        ax1.bar(df_stats['label'], df_stats['size'], color=self.cmap(np.linspace(0, 1, len(df_stats))))
        ax1.set_title('聚类大小分布')
        ax1.set_xlabel('聚类ID')
        ax1.set_ylabel('点数')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 聚类密度分布
        ax2 = axes[0, 1]
        ax2.scatter(df_stats['label'], df_stats['density'],
                    c=df_stats['size'], s=df_stats['size'] / 10,
                    cmap='viridis', alpha=0.7)
        ax2.set_title('聚类密度分布')
        ax2.set_xlabel('聚类ID')
        ax2.set_ylabel('密度')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 聚类半径分布
        ax3 = axes[0, 2]
        ax3.bar(df_stats['label'], df_stats['radius'], color=self.cmap(np.linspace(0, 1, len(df_stats))))
        ax3.set_title('聚类半径分布')
        ax3.set_xlabel('聚类ID')
        ax3.set_ylabel('半径')
        ax3.tick_params(axis='x', rotation=45)

        # 4. 聚类紧密度分布
        ax4 = axes[1, 0]
        ax4.scatter(df_stats['label'], df_stats['compactness'],
                    c=df_stats['size'], s=df_stats['size'] / 10,
                    cmap='viridis', alpha=0.7)
        ax4.set_title('聚类紧密度分布')
        ax4.set_xlabel('聚类ID')
        ax4.set_ylabel('紧密度')
        ax4.tick_params(axis='x', rotation=45)

        # 5. 凸包面积分布
        ax5 = axes[1, 1]
        ax5.bar(df_stats['label'], df_stats['convex_hull_area'], color=self.cmap(np.linspace(0, 1, len(df_stats))))
        ax5.set_title('凸包面积分布')
        ax5.set_xlabel('聚类ID')
        ax5.set_ylabel('面积')
        ax5.tick_params(axis='x', rotation=45)

        # 6. 聚类大小与半径的关系
        ax6 = axes[1, 2]
        scatter = ax6.scatter(df_stats['size'], df_stats['radius'],
                              c=df_stats['label'], s=df_stats['compactness'] * 10,
                              cmap=self.colormap, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax6.set_title('聚类大小 vs 半径')
        ax6.set_xlabel('大小')
        ax6.set_ylabel('半径')

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax6)
        cbar.set_label('聚类ID')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"统计分析图已保存到: {save_path}")

        return fig


def plot_trajectory_clusters(points: np.ndarray, labels: np.ndarray,
                             trajectories: Optional[List[np.ndarray]] = None,
                             title: str = "轨迹聚类可视化",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制轨迹聚类结果

    Args:
        points: 所有轨迹点
        labels: 聚类标签
        trajectories: 轨迹列表（可选）
        title: 图表标题
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    visualizer = ClusterVisualizer()

    if trajectories is not None:
        # 如果有轨迹数据，创建更复杂的可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 左侧：点聚类
        visualizer._plot_on_axis(points, labels, axes[0], "点聚类结果")

        # 右侧：轨迹聚类
        ax = axes[1]

        # 获取唯一的聚类标签
        unique_labels = np.unique(labels)
        colors = visualizer.cmap(
            np.linspace(0, 1, len(unique_labels) if 0 not in unique_labels else len(unique_labels) - 1))

        # 绘制每条轨迹
        for traj_idx, trajectory in enumerate(trajectories[:100]):  # 只显示前100条轨迹
            if len(trajectory) < 2:
                continue

            # 获取轨迹的主要聚类标签
            traj_points = trajectory
            traj_labels = []

            for point in traj_points:
                # 找到最近的点
                distances = np.linalg.norm(points - point, axis=1)
                nearest_idx = np.argmin(distances)
                traj_labels.append(labels[nearest_idx])

            # 使用最常见的标签作为轨迹标签
            if traj_labels:
                main_label = max(set(traj_labels), key=traj_labels.count)

                if main_label == 0:
                    color = 'gray'
                    alpha = 0.2
                    linewidth = 0.5
                else:
                    color = colors[main_label - 1 if 0 in unique_labels else main_label]
                    alpha = 0.6
                    linewidth = 1.0

                # 绘制轨迹
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                        color=color, alpha=alpha, linewidth=linewidth)

        ax.set_title("轨迹聚类结果")
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')

    else:
        # 只有点数据
        fig = visualizer.plot_clusters_2d(points, labels, title, save_path)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"轨迹聚类图已保存到: {save_path}")

    return fig


def plot_cluster_heatmap(points: np.ndarray, labels: np.ndarray,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    快速创建聚类热力图

    Args:
        points: 点数据
        labels: 聚类标签
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    visualizer = ClusterVisualizer()
    return visualizer.plot_cluster_heatmap(points, labels, save_path=save_path)


def plot_spatial_distribution(points: np.ndarray,
                              attribute: Optional[np.ndarray] = None,
                              title: str = "空间分布图",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制空间分布图

    Args:
        points: 点数据
        attribute: 属性数据（用于颜色映射）
        title: 图表标题
        save_path: 保存路径

    Returns:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    if attribute is not None:
        # 使用属性值进行颜色映射
        scatter = ax.scatter(points[:, 0], points[:, 1],
                             c=attribute, cmap='viridis',
                             s=10, alpha=0.6, edgecolors='w', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('属性值')
    else:
        # 简单的散点图
        ax.scatter(points[:, 0], points[:, 1],
                   s=5, alpha=0.5, color='blue', edgecolors='w', linewidth=0.2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'总点数: {len(points)}'
    if attribute is not None:
        stats_text += f'\n属性范围: [{np.min(attribute):.2f}, {np.max(attribute):.2f}]'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"空间分布图已保存到: {save_path}")

    return fig