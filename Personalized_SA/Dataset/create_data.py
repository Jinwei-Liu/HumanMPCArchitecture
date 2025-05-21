import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from Personalized_SA.env.quadrotor import *

def generate_gate_positions(base_positions, variability):
    num_gates = base_positions.shape[0]
    random_offsets = (np.random.rand(num_gates, 3) - 0.5) * 2 * variability
    actual_positions = base_positions + random_offsets
    return actual_positions

def plan_path(start_pos, gate_positions, end_pos, num_points, degree):
    waypoints = np.vstack((start_pos, gate_positions, end_pos))

    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]

    tck, u = splprep([x, y, z], s=0, k=degree)

    u_new = np.linspace(u.min(), u.max(), num_points)

    path_points = splev(u_new, tck)
    path_array = np.array(path_points).T 

    return path_array

def visualize_path_and_gates(start_pos, end_pos, gate_positions, path, true_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(gate_positions[:, 0], gate_positions[:, 1], gate_positions[:, 2],
               c='red', marker='s', s=100, label='Gates')
    for i, pos in enumerate(gate_positions):
         ax.text(pos[0], pos[1], pos[2] + 0.1, f'Gate {i+1}', color='red') 

    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='green', marker='o', s=100, label='Start')
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], c='blue', marker='x', s=100, label='End')

    ax.plot(path[:, 0], path[:, 1], path[:, 2], c='blue', linestyle='-', linewidth=1.5, label='Planned Path')
    ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2], c='red', linestyle='-', linewidth=1.5, label='True Path')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Drone Path Planning Through Gates')

    all_points = np.vstack((start_pos, end_pos, gate_positions, path))
    min_coords = np.min(all_points, axis=0) - 0.5
    max_coords = np.max(all_points, axis=0) + 0.5
    ax.set_xlim(min_coords[0], max_coords[0])
    ax.set_ylim(min_coords[1], max_coords[1])
    ax.set_zlim(min_coords[2], max_coords[2])

    ax.legend()

    plt.grid(True)
    plt.show()

import torch
def main():
    NUM_GATES = 4 
    START_POS = np.array([0, 0, 1])  
    END_POS = np.array([10, 0, 1])

    BASE_GATE_POSITIONS = np.array([
        [0.0, 0.5, 1.5],
        [4.0, -0.5, 1.0],
        [6.0, 0.5, 1.5],
        [15.0, -0.5, 1.0]
    ])

    GATE_POS_VARIABILITY = np.array([0.3, 0.3, 0.2]) # x, y, z 方向上的最大偏移量

    # 路径生成参数
    NUM_PATH_POINTS = 1000 # 生成路径点的数量 (离散化程度)
    SPLINE_DEGREE = 3     # B-样条曲线的次数 (通常用 3 次)

    actual_gate_positions = generate_gate_positions(BASE_GATE_POSITIONS, GATE_POS_VARIABILITY)

    print("--- 本次模拟生成的门位置 ---")
    for i, pos in enumerate(actual_gate_positions):
        print(f"门 {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print("-" * 28)

    planned_path = plan_path(START_POS, actual_gate_positions, END_POS, NUM_PATH_POINTS, SPLINE_DEGREE)

    DT          = 0.01          # 积分步长  (s)
    T_HORIZON   = 15         # MPC 预测步数

    quad = Quadrotor_MPC(DT)
    quad_env = Quadrotor_v0(DT)
    quad_env.reset()

    n_state, n_ctrl   = quad.s_dim, quad.a_dim

    w_pos, w_vel      = 1., 1e-5
    w_quat            = 1e-5
    w_act             = 1e-5
    n_batch           = 1

    goal_weights = torch.Tensor([w_pos, w_pos, w_pos,              # 位置
         w_quat, w_quat, w_quat, w_quat,   # 四元数 (w,x,y,z)
         w_vel, w_vel, w_vel]            # 速度
            )
    ctrl_weights = torch.Tensor([w_act,w_act,w_act,w_act])

    q = torch.cat((
    goal_weights,
    ctrl_weights
    ))

    C = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1, 1)
                     
    u_min = torch.tensor([0.0, -50.0, -20.0, -20.0])
    u_max = torch.tensor([100.0,  50.0,  20.0,  20.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)   # (25, 4)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)   # (25, 4)
    
    x = torch.zeros(n_state)
    x[kPosZ] = 1.0 # 注意修改
    x[kQuatW] = 1.0
    x = x.unsqueeze(0) 
    quad_env.set_state(x.squeeze(0).detach().cpu().numpy())

    x_history = []
    action_history = []
    x_goal_history = []
    steps = len(planned_path)
    print("steps:", steps)

    u_init = torch.tensor([0.0, 0.0, 0.0, 0.0])
    u_init = u_init.repeat(T_HORIZON, 1, 1)

    for step in range(steps):
        print(f"step: {step}")
        ctrl = mpc.MPC(n_state=n_state,
        n_ctrl=n_ctrl,
        T=T_HORIZON,
        u_lower=u_lower,
        u_upper=u_upper,
        u_init=u_init,
        prev_ctrl=u_init,
        lqr_iter=10,
        grad_method=GradMethods.ANALYTIC,
        exit_unconverged = False,
        eps=1,
        verbose=0)
        
        x_goal = torch.zeros(n_state)
        x_goal[kQuatW]    = 1.0                       # 悬停姿态
        x_goal[kPosX]    = planned_path[step, 0]        # 目标位置
        x_goal[kPosY]    = planned_path[step, 1]
        x_goal[kPosZ]    = planned_path[step, 2]

        px = -torch.sqrt(goal_weights)*x_goal
        p = torch.cat((px, torch.zeros(n_ctrl)))
        # x_aim =  torch.cat((x_goal, torch.tensor([9.81,0,0,0])))
        # p = -torch.sqrt(q)*x_aim

        c = p.unsqueeze(0).repeat(T_HORIZON, n_batch, 1)

        cost = QuadCost(C, c)  
        _, u_opt, _ = ctrl(x, cost, quad)
        u_init = u_opt
        action = u_opt[0,0].detach().cpu().numpy()
        state = quad_env.run(action)
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        x_history.append(state)
        action_history.append(action)
        x_goal_history.append(x_goal.detach().cpu().numpy())

    x_arr = np.stack(x_history)
    action_arr = np.stack(action_history)
    x_goal_arr = np.stack(x_goal_history)

    np.save('x_arr.npy', x_arr)
    np.save('action_arr.npy', action_arr)
    np.save('x_goal_arr.npy', x_goal_arr)

    visualize_path_and_gates(START_POS, END_POS, actual_gate_positions, planned_path, x_arr)

    fig, axes = plt.subplots(n_state,      # 行数 = 状态维度
                        1,            # 一列
                        sharex=True,  # 共用 x 轴
                        figsize=(6, 1.8*n_state))

    for dim in range(n_state):
        ax = axes[dim]                    # 当前子图
        ax.plot(x_arr[:, dim])            # 画出 dim 维
        ax.plot(x_goal_arr[:, dim])            # 画出 dim 维
        ax.set_ylabel(f"x[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Timestep")       # 只在最后一行标注
    fig.suptitle("Evolution of state vector x", y=1.02)
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(n_ctrl,      # 行数 = 控制维度
                    1,            # 一列
                    sharex=True,  # 共用 x 轴
                    figsize=(6, 1.8*n_ctrl))

    for dim in range(n_ctrl):
        ax = axes[dim]                    # 当前子图
        ax.plot(action_arr[:, dim])            # 画出 dim 维
        ax.set_ylabel(f"u[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Timestep")       # 只在最后一行标注
    fig.suptitle("Evolution of control vector u", y=1.02)
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()