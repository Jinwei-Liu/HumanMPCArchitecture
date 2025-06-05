import argparse
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Personalized_SA.human_model.rlhuman import test
from Personalized_SA.env.quadrotor import *
import torch.optim as optim
import torch.nn.utils as nn_utils

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--max_test_steps", type=int, default=2000)
parser.add_argument("--save_path", type=str, default="./Personalized_SA/human_model/checkpoints/actor.pth")
parser.add_argument("--load_model", type=str, default="./Personalized_SA/human_model/checkpoints/actor.pth")
args = parser.parse_args()

actions, states = test(args, temperature=1)
actions = np.array(actions)
states = np.array(states)

x_arr      = np.array(states)
action_arr = np.array(actions)
x_goal_arr = x_arr.copy() 

print("x_arr.shape:", x_arr.shape)         # (N, n_state)
print("action_arr.shape:", action_arr.shape)# (N, n_ctrl)
print("x_goal_arr.shape:", x_goal_arr.shape)# (N, n_state)

DT        = 0.01          # 积分步长 (s)
T_HORIZON = 15            # MPC 预测步数

quad      = Quadrotor_MPC(DT)
n_state   = quad.s_dim
n_ctrl    = quad.a_dim

# 2. 把整个数据集当作一个 batch 来跑
n_batch = x_arr.shape[0]     # 用全部样本做一个 batch

# 3. 把 numpy 转成 Torch 张量，并放到 GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_all      = torch.from_numpy(x_arr).float().to(device)      # [N, n_state]
action_all = torch.from_numpy(action_arr).float().to(device) # [N, n_ctrl]
# 注意：下面我们不把 x_goal_arr 直接用于计算，而是把它“包装”成一个可学习的参数
x_goal_init = torch.from_numpy(x_goal_arr).float().to(device) # [N, n_state]

# ----------------------------
# 4. 将 x_goal_init 变成一个 Parameter
# ----------------------------
# 原始 x_goal_arr 的每一行都是一个 target state (N, n_state)，希望把它当成可学习的参数
# 先 clone 一份，避免直接在原始数据上做操作
x_goal_param = torch.nn.Parameter(x_goal_init.clone(), requires_grad=True)  # [N, n_state]

# 5. 其他可学习的权重（与之前一致）
goal_weights = torch.ones(n_state, device=device) * 1e-2
goal_weights[0:3] = 0.5       # 前 3 维姿态/位置给稍大初值
goal_weights.requires_grad_(True)

ctrl_weights = torch.ones(n_ctrl, device=device) * 1e-2
ctrl_weights.requires_grad_(True)

# 6. MPC 的输入约束
u_min = torch.tensor([0.0, -50.0, -20.0, -20.0], device=device)
u_max = torch.tensor([100.0,  50.0,  20.0,  20.0], device=device)
# 这里要把约束也扩成 [T_HORIZON, batch_size, n_ctrl]
u_lower = u_min.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)  # [T_HORIZON, N, n_ctrl]
u_upper = u_max.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)  # [T_HORIZON, N, n_ctrl]

# 7. 构造两个不同的 Optimizer：
#    - optimizer_weights 只优化 goal_weights 和 ctrl_weights
#    - optimizer_goal    只优化 x_goal_param
optimizer_weights = optim.Adam([goal_weights, ctrl_weights], lr=0.01)
optimizer_goal    = optim.Adam([x_goal_param], lr=0.001)

# ----------------------------------------------------------------------------
# 8. 训练循环：在这里我们示范“偶数 epoch 更新 (goal_weights, ctrl_weights)；奇数 epoch 更新 x_goal_param”
# ----------------------------------------------------------------------------
for epoch in range(500):
    # 8.1 把 q_vector、C 一次性构造好，跟之前一样
    q_vector = torch.cat((goal_weights**2, ctrl_weights**2), dim=0)  # [n_state + n_ctrl]
    Q_diag = torch.diag(q_vector)                           # [n_state+n_ctrl, n_state+n_ctrl]
    C = Q_diag.unsqueeze(0).unsqueeze(0)                     # [1, 1, n_state+n_ctrl, n_state+n_ctrl]
    C = C.repeat(T_HORIZON, n_batch, 1, 1)                   # [T_HORIZON, N, n_state+n_ctrl, n_state+n_ctrl]

    # 8.2 构造 c 向量：注意这里要用 x_goal_param（因为它是当前要优化的目标）
    #     px = -sqrt(goal_weights^2) * x_goal_param
    px = -torch.sqrt(goal_weights**2).unsqueeze(0) * x_goal_param  # [N, n_state]
    zeros_u = torch.zeros((n_batch, n_ctrl), device=device)         # [N, n_ctrl]
    p_all   = torch.cat((px, zeros_u), dim=1)                       # [N, n_state+n_ctrl]
    c = p_all.unsqueeze(0).repeat(T_HORIZON, 1, 1)                 # [T_HORIZON, N, n_state+n_ctrl]

    # 8.3 x_batch 直接就是 x_all  (shape [N, n_state])
    x_batch = x_all

    # 8.4 构造 QuadCost
    cost = QuadCost(C, c)

    # 8.5 调用 MPC 求解一次 u_opt: [T_HORIZON, N, n_ctrl]
    ctrl = mpc.MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T_HORIZON,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=5,
        grad_method=GradMethods.ANALYTIC,
        exit_unconverged=False,
        detach_unconverged=False,
        verbose=0
    )
    _, u_opt, _ = ctrl(x_batch, cost, quad)  # u_opt: [T_HORIZON, N, n_ctrl]

    # 8.6 取第 0 步的动作，和真实动作对比，得到 loss
    u_pred = u_opt[0, :, :]               # [N, n_ctrl]
    delta_u = u_pred - action_all         # [N, n_ctrl]
    total_cost = torch.sum(delta_u**2)    # 标量

    # ----------------------------------------------------------------------------
    # 8.7 分开更新：偶数 epoch 更新 (goal_weights, ctrl_weights)，奇数 epoch 更新 x_goal_param
    #    **关键**：先把两个 Optimizer 里所有参数的 grad 都手动清零，以免梯度交叉累积
    # ----------------------------------------------------------------------------
    # 先统一清零（因为两个 Optimizer 里都可能含有 grad）
    optimizer_weights.zero_grad()
    optimizer_goal.zero_grad()

    if epoch % 2 == 0:
        # -----------------------------
        # 偶数 epoch：只更新 goal_weights 和 ctrl_weights
        # -----------------------------
        # 由于 x_goal_param 也在计算图里，但我们不希望它在这个步骤被更新，于是直接把它的 requires_grad 暂时设为 False
        # （也可以不改 requires_grad，只要不调用 optimizer_goal.step()，它就不会被更新；但为了让 x_goal_param 不计算梯度，可以这样做）
        x_goal_param.requires_grad_(False)

        total_cost.backward()  # 计算关于 goal_weights/ctrl_weights 的梯度
        # 可选做梯度裁剪
        nn_utils.clip_grad_norm_([goal_weights, ctrl_weights], max_norm=1.0)
        optimizer_weights.step()  # 只对 goal_weights、ctrl_weights 走 update

        # 做完之后再把 x_goal_param 恢复为可计算梯度
        x_goal_param.requires_grad_(True)

        print(f"Epoch {epoch:03d} | [更新 weights] total_cost = {total_cost.item():.6f}")
        print(f"  grad norm goal_weights = {goal_weights.grad.norm().item():.6f}")
        print(f"  grad norm ctrl_weights = {ctrl_weights.grad.norm().item():.6f}")
        print(f"  current goal_weights = {goal_weights.data.cpu().numpy()}")
        print(f"  current ctrl_weights = {ctrl_weights.data.cpu().numpy()}")

    else:
        # -----------------------------
        # 奇数 epoch：只更新 x_goal_param
        # -----------------------------
        # 由于 goal_weights/ctrl_weights 在计算图里，我们不想让它们计算梯度，可暂时把它们置为不需要 grad
        goal_weights.requires_grad_(False)
        ctrl_weights.requires_grad_(False)

        total_cost.backward()  # 计算关于 x_goal_param 的梯度
        # 可选做梯度裁剪（若有需要）：
        nn_utils.clip_grad_norm_([x_goal_param], max_norm=1.0)
        optimizer_goal.step()  # 只对 x_goal_param 走 update
        
        # 更新完后再把 weights 恢复为需要 grad
        goal_weights.requires_grad_(True)
        ctrl_weights.requires_grad_(True)

        print(f"Epoch {epoch:03d} | [更新 x_goal_arr] total_cost = {total_cost.item():.6f}")
        print(f"  grad norm x_goal_param = {x_goal_param.grad.norm().item():.6f}")
        # 如果想看 x_goal_param 的部分内容（比如前 5 行）：
        print(f"  current x_goal_param[:5] = {x_goal_param.data.cpu().numpy()[:5]}")

print("训练结束！")

# 假设 x_goal_param 是你想要绘制的目标参数，获取训练结束时的 x_goal_param
x_goal_final = x_goal_param.data.cpu().numpy()  # 获取推理出的目标状态数组
x_state = x_arr
# 创建一个 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图，假设每个点是 x_goal 的一个坐标（x, y, z）
ax.scatter(x_goal_final[:, 0], x_goal_final[:, 1], x_goal_final[:, 2], c='r', marker='o')
ax.scatter(x_state[:, 0], x_state[:, 1], x_state[:, 2], c='b', marker='o')

# 添加标题和轴标签
ax.set_title("Inference of x_goal during training (3D Scatter Plot)")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

# 展示图像
plt.show()