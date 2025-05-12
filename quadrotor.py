import numpy as np
from scipy.spatial.transform import Rotation as R
import torch, torch.nn as nn
#
from quad_index import *

#
class Quadrotor_v0(nn.Module):
    def __init__(self, dt):
        super().__init__()
        self.s_dim = 10
        self.a_dim = 4

        self._gz = 9.81
        self._dt = dt

    def forward(self, X, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        """
        # rk4 int
        M = 4
        DT = self._dt / M

        for _ in range(M):
            k1 = DT*self._f(X, action)
            k2 = DT*self._f(X + 0.5*k1, action)
            k3 = DT*self._f(X + 0.5*k2, action)
            k4 = DT*self._f(X + k3, action)
            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0

        return X

    def _f(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        thrust, wx, wy, wz = action.split(1, dim=-1)

        dstate = torch.zeros_like(state)

        dstate[:,kPosX:kPosZ+1] = state[:,kVelX:kVelZ+1]


        quat = state[...,kQuatW:kQuatZ+1]
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)

        qw, qx, qy, qz = quat.split(1, dim=-1)

        dstate[:,kQuatW] = 0.5 * ( -wx*qx - wy*qy - wz*qz )
        dstate[:,kQuatX] = 0.5 * (  wx*qw + wz*qy - wy*qz )
        dstate[:,kQuatY] = 0.5 * (  wy*qw - wz*qx + wx*qz )
        dstate[:,kQuatZ] = 0.5 * (  wz*qw + wy*qx - wx*qy )

        dstate[:,kVelX] = 2 * ( qw*qy + qx*qz ) * thrust
        dstate[:,kVelY] = 2 * ( qy*qz - qw*qx ) * thrust
        dstate[:,kVelZ] = (qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz

        return dstate

from torchviz import make_dot
def main():
    dt = 0.01
    quadrotor = Quadrotor_v0(dt)

    initial_state = quadrotor.reset()  
    print("Initial State:", initial_state)

    action = torch.tensor([5.0, 0.1, 0.2, 0.3])

    next_state = quadrotor.forward(initial_state, action)
    print("Next State after applying action:", next_state)

    loss = torch.sum(next_state) 
    print(loss)
    
    dot = make_dot(loss)
    dot.format = 'pdf'
    dot.render('computation_graph')

from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
import matplotlib.pyplot as plt
def test_mpc():
    # ------------ 关键超参数 ------------
    DT          = 0.02          # 积分步长  (s)
    T_HORIZON   = 5         # MPC 预测步数
    step = 200

    # ---------- 初始化 ----------
    quad = Quadrotor_v0(DT)

    n_state, n_ctrl   = quad.s_dim, quad.a_dim

    # ----------- 1. 目标状态 & 权重 -----------------
    x_goal            = torch.zeros(n_state)
    x_goal[kQuatW]    = 1.0                       # 悬停姿态

    w_pos, w_vel      = 100., 50.
    w_quat            = 10.
    w_act             = 0.001
    n_batch           = 1

    # ----------- 2. 打包成 C, c --------------------
    goal_weights = torch.Tensor([w_pos, w_pos, w_pos,              # 位置
         w_quat, w_quat, w_quat, w_quat,   # 四元数 (w,x,y,z)
         w_vel, w_vel, w_vel]            # 速度
            )
    ctrl_weights = torch.Tensor([w_act,w_act,w_act,w_act])

    q = torch.cat((
    goal_weights,
    ctrl_weights
    ))

    px = -torch.sqrt(goal_weights)*x_goal

    p = torch.cat((px, torch.zeros(n_ctrl)))

    # 按时间 & batch 维度 broadcast
    C = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1, 1)
    c = p.unsqueeze(0).repeat(T_HORIZON, n_batch, 1)

    cost = QuadCost(C, c)                         

    # ----------- 3. 控制器 -------------------------
    u_min = torch.tensor([0.0, -2.0, -2.0, -2.0])
    u_max = torch.tensor([20.0,  2.0,  2.0,  2.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)   # (25, 4)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)   # (25, 4)

    ctrl = mpc.MPC(n_state=n_state,
                   n_ctrl=n_ctrl,
                   T=T_HORIZON,
                   u_lower=u_lower,
                   u_upper=u_upper,
                   lqr_iter=10,
                   grad_method=GradMethods.AUTO_DIFF,
                   exit_unconverged = False,
                   verbose=0)

    # ----------- 4. 主循环 -------------------------
    x = torch.zeros(n_state)
    x[kQuatW] = 1.0
    x[kPosZ] = -1.0
    x = x.unsqueeze(0) 
    x_history = []
    for i in range(step):
        print(i)
        _, u_opt, _ = ctrl(x, cost, quad)
        print(u_opt[0])
        x = quad(x,u_opt[0])
        x_history.append(x.squeeze(0).detach().cpu().numpy())

    x_arr = np.stack(x_history)
    step, n_state = x_arr.shape
    # === 2. 创建子图 =====================
    fig, axes = plt.subplots(n_state,      # 行数 = 状态维度
                            1,            # 一列
                            sharex=True,  # 共用 x 轴
                            figsize=(6, 1.8*n_state))

    for dim in range(n_state):
        ax = axes[dim]                    # 当前子图
        ax.plot(x_arr[:, dim])            # 画出 dim 维
        ax.set_ylabel(f"x[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Timestep")       # 只在最后一行标注
    fig.suptitle("Evolution of state vector x", y=1.02)
    fig.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    test_mpc()