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

# # ─────────────────────────────────────────
# # 1. 随机生成示例数据
# # ─────────────────────────────────────────
# np.random.seed(42)        

# # (1) 真实轨迹：3D 随机游走 (250 帧)
# N = 500
# state_steps = np.random.normal(scale=0.6, size=(N, 3))
# state_array = np.cumsum(state_steps, axis=0)

# # (2) 预测轨迹：对每隔 20 帧的时刻，向前预测 15 步
# idx  = np.arange(0, N, 20)      # 预测起点索引
# M, T = len(idx), 15
# store_predict_array = np.zeros((M, T, 3))
# for m, id0 in enumerate(idx):
#     base       = state_array[id0]
#     direction  = np.random.randn(3)
#     direction /= np.linalg.norm(direction)          # 归一化方向
#     steps      = np.linspace(0.4, 6.5, T).reshape(-1, 1)
#     store_predict_array[m] = base + direction * steps \
#                              + np.random.normal(scale=0.2, size=(T, 3))

# # (3) 控制轨迹：在预测值基础上再加少量噪声
# hold_u_x_array = store_predict_array \
#                  + np.random.normal(scale=0.3, size=store_predict_array.shape)

# # (4) 目标点：随便放一个
# aim_goal_array = np.array([[10, 10, 10]])

# # ─────────────────────────────────────────
# # 2. 画图，检验函数是否工作正常
# # ─────────────────────────────────────────
# plot_state_3d(state_array,
#               store_predict_array,
#               hold_u_x_array,
#               [[0,0,0], [5,5,5], [10,10,10]],  # 假设的门位置
#               save_path=None,        # 想保存就写 "demo_3d.pdf" / "demo_3d.png"
#               show=True)