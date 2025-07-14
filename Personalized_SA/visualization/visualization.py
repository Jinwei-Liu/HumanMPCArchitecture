import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_state_3d(state_array, 
                  store_predict_array,
                  hold_u_x_array,
                  gate_positions,
                  figsize_mm=(83, 70),
                  dpi=300,
                  elev=90, azim=-90,
                  save_path=None,
                  show=True):
    """
    Nature/Science 风格的 3D 轨迹散点图（带连线版）
    """
    # -------- 1. 纯 ASCII 的 rcParams --------
    rc_params = {
        "font.family"      : "Arial",
        "font.size"        : 7.5,
        "axes.linewidth"   : 0.6,
        "axes.labelsize"   : 8,
        "axes.labelpad"    : 2,
        "xtick.major.size" : 3,
        "xtick.major.width": 0.5,
        "ytick.major.size" : 3,
        "ytick.major.width": 0.5,
    }
    mpl.rcParams.update(rc_params)

    # -------- 2. 颜色 --------
    c_obs, c_pred, c_control = "#4C72B0", "#DD8452", "#55A868"
    c_gate = "#7F7F7F"
    c_face = "#FFFFFF"

    # -------- 3. 数据整理 --------
    state_array         = np.asarray(state_array)
    store_predict_array = np.asarray(store_predict_array)
    hold_u_x_array      = np.asarray(hold_u_x_array)

    # -------- 4. 画布 --------
    w, h = figsize_mm
    fig = plt.figure(figsize=(w/25.4, h/25.4), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")

    # 观测轨迹（散点 + 连线）
    ax.scatter(state_array[:, 0], state_array[:, 1], state_array[:, 2],
               c=c_obs, s=7, marker="o", edgecolors="none", alpha=0.85,
               label="Observed")
    ax.plot(state_array[:, 0], state_array[:, 1], state_array[:, 2],
            c=c_obs, lw=0.7, alpha=0.9)

    # # 预测轨迹（散点 + 连线）
    # ax.scatter(store_predict_array[::20, :15, 0].ravel(),
    #            store_predict_array[::20, :15, 1].ravel(),
    #            store_predict_array[::20, :15, 2].ravel(),
    #            c=c_pred, s=6, marker="^", edgecolors="none", alpha=0.70,
    #            label="Predicted")
    # # 对每一条预测序列画线
    # for traj in store_predict_array[::20, :15]:
    #     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
    #             c=c_pred, lw=0.6, alpha=0.65)

    # # 控制轨迹（可选：散点 + 连线）
    # ax.scatter(hold_u_x_array[::20, :15, 0].ravel(),
    #             hold_u_x_array[::20, :15, 1].ravel(),
    #             hold_u_x_array[::20, :15, 2].ravel(),
    #             c=c_control, s=6, marker="s", edgecolors="none", alpha=0.65,
    #             label="Control")
    # for traj in hold_u_x_array[::20, :15]:
    #     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
    #             c=c_control, lw=0.6, alpha=0.55)
    
    # -------- 添加门 --------
    if len(gate_positions) > 0:
        # 门的尺寸参数
        gate_width = 1.5
        gate_height = 1.5
        gate_depth = 0.5

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # 绘制每个门
        for gate_pos in gate_positions:
            x, y, z = gate_pos
            
            # 创建门的顶点 - 垂直立方体
            vertices = [
                # 前面
                [x-gate_width/2, y-gate_depth/2, z-gate_height/2],
                [x+gate_width/2, y-gate_depth/2, z-gate_height/2],
                [x+gate_width/2, y-gate_depth/2, z+gate_height/2],
                [x-gate_width/2, y-gate_depth/2, z+gate_height/2],
                # 后面
                [x-gate_width/2, y+gate_depth/2, z-gate_height/2],
                [x+gate_width/2, y+gate_depth/2, z-gate_height/2],
                [x+gate_width/2, y+gate_depth/2, z+gate_height/2],
                [x-gate_width/2, y+gate_depth/2, z+gate_height/2]
            ]
            
            # 定义每个面的顶点索引
            faces = [
                [0, 1, 2, 3],  # 前面
                [4, 5, 6, 7],  # 后面
                [0, 3, 7, 4],  # 左面
                [1, 2, 6, 5],  # 右面
                [0, 1, 5, 4],  # 下面
                [3, 2, 6, 7]   # 上面
            ]
            
            # 创建3D多边形集合
            poly3d = [[vertices[idx] for idx in face] for face in faces]
            gate = Poly3DCollection(poly3d, alpha=0.1, linewidths=1, edgecolors=c_gate)
            gate.set_facecolor(c_face)
            ax.add_collection3d(gate)
            
            # 添加门框架的线条以增强立体感
            edges = [
                # 垂直边
                [0, 3], [1, 2], [4, 7], [5, 6],
                # 水平边
                [0, 1], [3, 2], [4, 5], [7, 6],
                # 深度边
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            for edge in edges:
                ax.plot3D(
                    [vertices[edge[0]][0], vertices[edge[1]][0]],
                    [vertices[edge[0]][1], vertices[edge[1]][1]],
                    [vertices[edge[0]][2], vertices[edge[1]][2]],
                    color=c_gate, linewidth=1
                )

    # -------- 5. 坐标轴 --------
    ax.set_xlim(-10, 10); ax.set_ylim(-1, 20); ax.set_zlim(-10, 10)
    ax.set_xlabel("X position"); ax.set_ylabel("Y position"); ax.set_zlabel("Z position")

    try: ax.set_box_aspect((1, 1, 1))
    except AttributeError: pass

    ax.view_init(elev=elev, azim=azim)

    # 去掉面板背景
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("w")
        pane.set_facecolor((1,1,1,0))

    # 可选：手动设置 z 轴刻度线宽度
    for t in ax.zaxis.get_major_ticks():
        t.tick1line.set_markeredgewidth(0.5)
        t.tick2line.set_markeredgewidth(0.5)
        t.tick1line.set_markersize(3)
        t.tick2line.set_markersize(3)

    ax.legend(frameon=False, loc="upper right", fontsize=7)
    plt.tight_layout(pad=0.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Personalized_SA.config.config import args

def create_comparison_plot(threshold_vel=1):
    # 读取三个CSV文件
    file_names = [
        f'./Personalized_SA/visualization/threshold_vel_{threshold_vel}_hold_u_x_metrics.csv',
        f'./Personalized_SA/visualization/threshold_vel_{threshold_vel}_store_predict_metrics.csv',
        f'./Personalized_SA/visualization/threshold_vel_{threshold_vel}_nn_metrics.csv'
    ]
    
    # 读取数据
    dataframes = []
    labels = ['Hold U X', 'Store Predict', 'NN']
    
    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)
            dataframes.append(df)
        except FileNotFoundError:
            print(f"文件 {file_name} 未找到")
            return
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # 定义步数和状态
    steps = [5, 10, 20, 30, 40, 50]
    states_to_plot = ['x', 'y', 'z', 'qw', 'qx', 'qy','qz','vx','vy','vz']  # 选择10个状态进行绘图
    
    # 创建子图 - 为legend留出空间
    fig, axes = plt.subplots(1, 10, figsize=(22, 3))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 三种不同颜色
    
    # 用于存储legend的handles和labels
    legend_handles = []
    legend_labels = []
    
    for idx, state in enumerate(states_to_plot):
        ax = axes[idx]
        
        # 为每个数据源绘制线条
        for df_idx, (df, label) in enumerate(zip(dataframes, labels)):
            # 提取该状态的数据
            state_data = df[df['State'] == state]
            
            if len(state_data) == 0:
                continue
                
            means = []
            stds = []
            
            # 提取每个步数的均值和标准差
            for step in steps:
                mean_col = f'{step}_steps_mean'
                std_col = f'{step}_steps_std'
                
                if mean_col in state_data.columns and std_col in state_data.columns:
                    means.append(state_data[mean_col].iloc[0])
                    stds.append(state_data[std_col].iloc[0])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            
            # 绘制均值线和误差带
            line = ax.plot(steps, means, 'o-', label=label, color=colors[df_idx], 
                          linewidth=2, markersize=6)
            # ax.fill_between(steps, means - stds, means + stds, 
            #                alpha=0.2, color=colors[df_idx])
            
            # 只在第一个子图时收集legend信息
            if idx == 0:
                legend_handles.append(line[0])
                legend_labels.append(label)
        
        # 设置子图属性
        ax.set_xlabel('Steps', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.set_title(f'{state}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # 移除每个子图的legend
        # ax.legend(fontsize=10)
        
        # 设置x轴刻度
        ax.set_xticks(steps)
        ax.set_xticklabels(steps, fontsize=8)
        
        # 调整y轴标签字体大小
        ax.tick_params(axis='y', labelsize=8)
    
    # 添加全局legend
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.95),
               ncol=3, 
               fontsize=12,
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # 调整布局 - 为legend和title留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)  # 为legend留出空间
    
    # 添加主标题
    plt.suptitle(f'Comparison of Different Methods (Threshold Velocity = {threshold_vel})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图片
    store_csv_path = args.visualization_save_path.replace('.npz', '_comparison.png')
    plt.savefig(store_csv_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 使用函数
    create_comparison_plot(threshold_vel=args.threshold_vel)
