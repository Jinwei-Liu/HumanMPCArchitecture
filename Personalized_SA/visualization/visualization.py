import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_state_3d(state_array,
                  store_predict_array,
                  hold_u_x_array=None,
                  figsize_mm=(83, 70),
                  dpi=300,
                  elev=25, azim=135,
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

    # -------- 3. 数据整理 --------
    state_array         = np.asarray(state_array)
    store_predict_array = np.asarray(store_predict_array)
    hold_u_x_array  = np.asarray(hold_u_x_array)

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

    # 预测轨迹（散点 + 连线）
    ax.scatter(store_predict_array[::20, :15, 0].ravel(),
               store_predict_array[::20, :15, 1].ravel(),
               store_predict_array[::20, :15, 2].ravel(),
               c=c_pred, s=6, marker="^", edgecolors="none", alpha=0.70,
               label="Predicted")
    # 对每一条预测序列画线
    for traj in store_predict_array[::20]:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                c=c_pred, lw=0.6, alpha=0.65)

    # 控制轨迹（可选：散点 + 连线）
    if hold_u_x_array is not None:
        ax.scatter(hold_u_x_array[::20, :15, 0].ravel(),
                   hold_u_x_array[::20, :15, 1].ravel(),
                   hold_u_x_array[::20, :15, 2].ravel(),
                   c=c_control, s=6, marker="s", edgecolors="none", alpha=0.65,
                   label="Control")
        for traj in hold_u_x_array[::20]:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    c=c_control, lw=0.6, alpha=0.55)

    # -------- 5. 坐标轴 --------
    ax.set_xlim(-10, 10); ax.set_ylim(-5, 20); ax.set_zlim(-5, 20)
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
# ─────────────────────────────────────────
# 1. 随机生成示例数据
# ─────────────────────────────────────────
np.random.seed(42)        

# (1) 真实轨迹：3D 随机游走 (250 帧)
N = 500
state_steps = np.random.normal(scale=0.6, size=(N, 3))
state_array = np.cumsum(state_steps, axis=0)

# (2) 预测轨迹：对每隔 20 帧的时刻，向前预测 15 步
idx  = np.arange(0, N, 20)      # 预测起点索引
M, T = len(idx), 15
store_predict_array = np.zeros((M, T, 3))
for m, id0 in enumerate(idx):
    base       = state_array[id0]
    direction  = np.random.randn(3)
    direction /= np.linalg.norm(direction)          # 归一化方向
    steps      = np.linspace(0.4, 6.5, T).reshape(-1, 1)
    store_predict_array[m] = base + direction * steps \
                             + np.random.normal(scale=0.2, size=(T, 3))

# (3) 控制轨迹：在预测值基础上再加少量噪声
hold_u_x_array = store_predict_array \
                 + np.random.normal(scale=0.3, size=store_predict_array.shape)

# (4) 目标点：随便放一个
aim_goal_array = np.array([[10, 10, 10]])

# ─────────────────────────────────────────
# 2. 画图，检验函数是否工作正常
# ─────────────────────────────────────────
plot_state_3d(state_array,
              store_predict_array,
              hold_u_x_array=hold_u_x_array,
              save_path=None,        # 想保存就写 "demo_3d.pdf" / "demo_3d.png"
              show=True)