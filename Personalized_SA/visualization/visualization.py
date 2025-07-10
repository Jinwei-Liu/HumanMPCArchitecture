import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_state_3d(state_array,
                  store_predict_array,
                  hold_u_x_array=None,
                  aim_goal_array=None,
                  figsize_mm=(83, 70),
                  dpi=300,
                  elev=25, azim=135,
                  save_path=None,
                  show=True):
    """
    Nature/Science 风格的 3D 轨迹散点图
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
    c_obs, c_pred, c_control, c_goal = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

    # -------- 3. 数据整理 --------
    state_array         = np.asarray(state_array)
    store_predict_array = np.asarray(store_predict_array)
    if hold_u_x_array is not None:
        hold_u_x_array  = np.asarray(hold_u_x_array)
    if aim_goal_array is not None:
        aim_goal_array  = np.asarray(aim_goal_array)

    # -------- 4. 画布 --------
    w, h = figsize_mm
    fig = plt.figure(figsize=(w/25.4, h/25.4), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")

    # 观测轨迹
    ax.scatter(state_array[:, 0], state_array[:, 1], state_array[:, 2],
               c=c_obs, s=7, marker="o", edgecolors="none", alpha=0.85,
               label="Observed")

    # 预测轨迹
    ax.scatter(store_predict_array[::20, :15, 0].ravel(),
               store_predict_array[::20, :15, 1].ravel(),
               store_predict_array[::20, :15, 2].ravel(),
               c=c_pred, s=6, marker="^", edgecolors="none", alpha=0.70,
               label="Predicted")

    # 控制轨迹
    if hold_u_x_array is not None:
        ax.scatter(hold_u_x_array[::20, :15, 0].ravel(),
                   hold_u_x_array[::20, :15, 1].ravel(),
                   hold_u_x_array[::20, :15, 2].ravel(),
                   c=c_control, s=6, marker="s", edgecolors="none", alpha=0.65,
                   label="Control")

    # 目标点
    if aim_goal_array is not None:
        ax.scatter(aim_goal_array[:, 0], aim_goal_array[:, 1], aim_goal_array[:, 2],
                   c=c_goal, s=18, marker="*", edgecolors="k", linewidth=0.3,
                   label="Goal")

    # -------- 5. 坐标轴 --------
    ax.set_xlim(-5, 20); ax.set_ylim(-5, 20); ax.set_zlim(-5, 20)
    ax.set_xlabel("X position"); ax.set_ylabel("Y position"); ax.set_zlabel("Z position")

    try: ax.set_box_aspect((1, 1, 1))
    except AttributeError: pass

    ax.view_init(elev=elev, azim=azim)

    # 去掉面板背景
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("w")
        pane.set_facecolor((1,1,1,0))
    ax.grid(False)

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
