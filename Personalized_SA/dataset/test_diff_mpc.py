import sys
import torch.optim as optim
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from Personalized_SA.env.quadrotor import *
from torchviz import make_dot
import torch.nn.utils as nn_utils

def main():
    x_arr = np.load('x_arr.npy')
    action_arr = np.load('action_arr.npy')
    x_goal_arr = np.load('x_goal_arr.npy')
    print(x_arr.shape)
    print(action_arr.shape)
    print(x_goal_arr.shape)

    DT          = 0.01          # 积分步长  (s)
    T_HORIZON   = 15         # MPC 预测步数

    quad = Quadrotor_MPC(DT)

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
                     
    u_min = torch.tensor([0.0, -20.0, -20.0, -20.0])
    u_max = torch.tensor([100.0,  20.0,  20.0,  20.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)   # (25, 4)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)   # (25, 4)

    steps = len(x_arr)

    x = torch.from_numpy(x_arr[100]).float()
    x = x.unsqueeze(0)
    print("x:", x)
    action = torch.from_numpy(action_arr[100]).float()
    action = action.unsqueeze(0)
    x_goal_true = torch.from_numpy(x_goal_arr[100]).float()
    print("x_goal_true:", x_goal_true)

    x_goal = torch.zeros(n_state)
    x_goal[kQuatW]    = 1.0                       # 悬停姿态
    x_goal[kPosZ]    = 1.0                       # 悬停高度
    x_goal.requires_grad_(True)

    for _ in range(10):
        px = -torch.sqrt(goal_weights)*x_goal
        p = torch.cat((px, torch.zeros(n_ctrl)))
        c = p.unsqueeze(0).repeat(T_HORIZON, n_batch, 1)
        
        cost = QuadCost(C, c)  
        ctrl = mpc.MPC(n_state=n_state,
        n_ctrl=n_ctrl,
        T=T_HORIZON,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=20,
        grad_method=GradMethods.ANALYTIC,
        exit_unconverged=False,
        detach_unconverged=False,
        # eps=1,
        # back_eps=1,
        verbose=0)
        
        _, u_opt, _ = ctrl(x, cost, quad)
        print("u_opt:", u_opt[0])
        print("action:", action)
        print("x_goal:", x_goal)
        # 
        # Assuming `u_opt[0]` and `action` are given
        delta_u = u_opt[0] - action  # Calculate the difference between the optimal input and the actual action

        # Define a simple cost function based on the difference between u_opt[0] and action
        cost_function = torch.sum(delta_u**2)  # Simple L2 norm cost for control input difference
        print("Cost function value:", cost_function.item())

        dot = make_dot(cost_function, params={'x_goal': x_goal})
        dot.format = 'pdf'
        dot.render('computation_graph')

        # Perform optimization to update x_goal based on this cost
        optimizer = optim.Adam([x_goal], lr=0.01)  # Using Adam optimizer for gradient descent

        # Zero the gradients from the previous step
        optimizer.zero_grad()

        # Backpropagate the cost to compute the gradients of x_goal
        cost_function.backward()
        # Update the x_goal using the computed gradients
        optimizer.step()

        # Print the updated x_goal
        print("Updated x_goal:", x_goal)

def main2():
    DT          = 0.01          # 积分步长  (s)
    T_HORIZON   = 15         # MPC 预测步数

    quad = Quadrotor_MPC(DT)

    n_state, n_ctrl   = quad.s_dim, quad.a_dim

    w_pos, w_vel      = 1., 0.0001
    w_quat            = 0.0001
    w_act             = 0.00001
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
                     
    u_min = torch.tensor([0.0, -20.0, -20.0, -20.0])
    u_max = torch.tensor([50.0,  20.0,  20.0,  20.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)   # (25, 4)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)   # (25, 4)

    x = torch.tensor([0.0000,  0.0000,  0.0004,  0.9995, -0.0241,  0.0197,  0.0000,  0.0000, 0.0000,  0.0578])
    x = x.unsqueeze(0)
    action = torch.tensor([10.4218,  0.0000,  0.0000,  0.0000])
    action = action.unsqueeze(0)

    x_goal = torch.zeros(n_state)
    x_goal[kQuatW]    = 1.0                       # 悬停姿态
    x_goal[kPosZ]    = 1.0                       # 悬停高度
    x_goal.requires_grad_(True)

    px = -torch.sqrt(goal_weights)*x_goal
    p = torch.cat((px, torch.zeros(n_ctrl)))
    c = p.unsqueeze(0).repeat(T_HORIZON, n_batch, 1)
    
    cost = QuadCost(C, c)  
    ctrl = mpc.MPC(n_state=n_state,
    n_ctrl=n_ctrl,
    T=T_HORIZON,
    u_lower=u_lower,
    u_upper=u_upper,
    lqr_iter=30,
    eps=1e-2,
    slew_rate_penalty=1e-2,
    back_eps=1e-2,
    grad_method=GradMethods.ANALYTIC,
    exit_unconverged=False,
    verbose=1)

    _, u_opt, _ = ctrl(x, cost, quad)
    print("u_opt:", u_opt[0])
    print("action:", action)
    print("x_goal:", x_goal)
    # 
    # Assuming `u_opt[0]` and `action` are given
    delta_u = u_opt[0] - action  # Calculate the difference between the optimal input and the actual action

    # Define a simple cost function based on the difference between u_opt[0] and action
    cost_function = torch.sum(delta_u**2)  # Simple L2 norm cost for control input difference
    print("Cost function value:", cost_function.item())

    dot = make_dot(cost_function, params={'x_goal': x_goal})
    dot.format = 'pdf'
    dot.render('computation_graph')

    # Perform optimization to update x_goal based on this cost
    optimizer = optim.Adam([x_goal], lr=0.01)  # Using Adam optimizer for gradient descent

    # Zero the gradients from the previous step
    optimizer.zero_grad()

    # Backpropagate the cost to compute the gradients of x_goal
    cost_function.backward()
    # Update the x_goal using the computed gradients
    optimizer.step()

    # Print the updated x_goal
    print("Updated x_goal:", x_goal)

def main3():
    x_arr = np.load('x_arr.npy')
    action_arr = np.load('action_arr.npy')
    x_goal_arr = np.load('x_goal_arr.npy')
    print(x_arr.shape)
    print(action_arr.shape)
    print(x_goal_arr.shape)

    DT = 0.01          # 积分步长  (s)
    T_HORIZON = 15     # MPC 预测步数

    quad = Quadrotor_MPC(DT)

    n_state, n_ctrl = quad.s_dim, quad.a_dim
    
    n_batch = 1 

    # Initialize learnable weights
    goal_weights = torch.ones(10) *1e-2 # Initialize with small values
    goal_weights[0:3] = 0.5 # Position weights start higher
    goal_weights.requires_grad_(True)
    
    ctrl_weights = torch.ones(4) *1e-2 # Initialize with small values
    ctrl_weights.requires_grad_(True)

    u_min = torch.tensor([0.0, -50.0, -20.0, -20.0])
    u_max = torch.tensor([100.0,  50.0,  20.0,  20.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)

    optimizer = optim.Adam([goal_weights, ctrl_weights], lr=0.01)

    for epoch in range(200):  # Number of epochs for training
        total_cost = 0.0  # To accumulate cost over all samples
        for i in range(len(x_arr)):
            x = torch.from_numpy(x_arr[i]).float().unsqueeze(0)
            action = torch.from_numpy(action_arr[i]).float().unsqueeze(0)
            x_goal_true = torch.from_numpy(x_goal_arr[i]).float()

            # x_goal = torch.zeros(n_state)
            # x_goal[kQuatW] = 1.0                       # 悬停姿态
            # x_goal[kPosZ] = 1.0                        # 悬停高度
            x_goal = x_goal_true

            # Create weight diagonal matrix from learnable weights
            q = torch.cat((goal_weights**2, ctrl_weights**2))
            C = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1, 1)

            px = -torch.sqrt(goal_weights**2) * x_goal
            p = torch.cat((px, torch.zeros(n_ctrl)))
            c = p.unsqueeze(0).repeat(T_HORIZON, n_batch, 1)
            
            cost = QuadCost(C, c)  
            ctrl = mpc.MPC(n_state=n_state,
                          n_ctrl=n_ctrl,
                          T=T_HORIZON,
                          u_lower=u_lower,
                          u_upper=u_upper,
                          lqr_iter=5,
                          grad_method=GradMethods.ANALYTIC,
                          exit_unconverged=False,
                          detach_unconverged=False,
                          verbose=0)

            _, u_opt, _ = ctrl(x, cost, quad)
            # print("u_opt:", u_opt[0])
            # print("action:", action)
            
            # Calculate the difference between the optimal input and the actual action
            delta_u = u_opt[0] - action

            # Define cost function based on the difference
            cost = torch.sum(delta_u**2)
            
            total_cost += cost  # Accumulate cost

        print("Epoch:", epoch)
        print("total_cost value:", total_cost.item())
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Backpropagate the total cost
        total_cost.backward()
        nn_utils.clip_grad_norm_([goal_weights, ctrl_weights], max_norm=1.0)
        print(' goal_weights.grad  =', goal_weights.grad)        # 可用 .norm() 查看幅值
        print(' ctrl_weights.grad  =', ctrl_weights.grad)
        # Update the weights using the computed gradients
        optimizer.step()

        print("Current goal_weights:", goal_weights)
        print("Current ctrl_weights:", ctrl_weights)

def main4():
    x_arr        = np.load('x_arr.npy')        # 形状 (N, n_state)
    action_arr   = np.load('action_arr.npy')   # 形状 (N, n_ctrl)
    x_goal_arr   = np.load('x_goal_arr.npy')   # 形状 (N, n_state)

    print("x_arr.shape:", x_arr.shape)         # (N, n_state)
    print("action_arr.shape:", action_arr.shape)# (N, n_ctrl)
    print("x_goal_arr.shape:", x_goal_arr.shape)# (N, n_state)

    DT        = 0.01          # 积分步长 (s)
    T_HORIZON = 15            # MPC 预测步数

    quad      = Quadrotor_MPC(DT)
    n_state   = quad.s_dim
    n_ctrl    = quad.a_dim

    # 2. 这里直接把整个数据集当作一个 batch 来跑
    #    如果数据集很大，也可以自己设定一个较小的 n_batch 然后在外面循环分批次
    n_batch = x_arr.shape[0]     # 用全部样本做一个 batch
    # --- 如果想做 mini-batch，可以把这里设成一个小于 N 的值，然后下面写一个 for b_i in range(0, N, n_batch) 来分批次处理 ---

    # 把 numpy 先转换成 torch.Tensor，注意放到 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_all      = torch.from_numpy(x_arr).float().to(device)      # [N, n_state]
    action_all = torch.from_numpy(action_arr).float().to(device) # [N, n_ctrl]
    x_goal_all = torch.from_numpy(x_goal_arr).float().to(device) # [N, n_state]

    # 3. 可学习的权重
    #    比如 goal_weights 长度是 n_state，ctrl_weights 长度是 n_ctrl
    goal_weights = torch.ones(n_state) * 1e-2
    goal_weights[0:3] = 0.5       # 前 3 维姿态/位置给稍大初值
    goal_weights = goal_weights.to(device)
    goal_weights.requires_grad_(True)

    ctrl_weights = torch.ones(n_ctrl) * 1e-2
    ctrl_weights = ctrl_weights.to(device)
    ctrl_weights.requires_grad_(True)

    # MPC 的输入约束
    u_min = torch.tensor([0.0, -50.0, -20.0, -20.0], device=device)
    u_max = torch.tensor([100.0,  50.0,  20.0,  20.0], device=device)
    # 这里要把约束也扩成 [T_HORIZON, batch_size, n_ctrl]
    u_lower = u_min.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)  # [T_HORIZON, batch, n_ctrl]
    u_upper = u_max.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)  # [T_HORIZON, batch, n_ctrl]

    optimizer = optim.Adam([goal_weights, ctrl_weights], lr=0.01)

    # 4. 训练循环
    for epoch in range(200):
        # 4.1 构造一次性用于整个 batch 的 C 和 c
        #     q = [goal_weights**2, ctrl_weights**2]，长度为 n_state+n_ctrl
        q_vector = torch.cat((goal_weights**2, ctrl_weights**2), dim=0)  # [n_state + n_ctrl]
        #     把 q_vector 放到对角矩阵上，并 repeat 到 [T_HORIZON, n_batch, n_state+n_ctrl, n_state+n_ctrl]
        #     注意：QuadCost 每一步用到的是 state+ctrl 的联合二次形式
        Q_diag = torch.diag(q_vector)                           # [n_state+n_ctrl, n_state+n_ctrl]
        C = Q_diag.unsqueeze(0).unsqueeze(0)                     # [1, 1, n_state+n_ctrl, n_state+n_ctrl]
        C = C.repeat(T_HORIZON, n_batch, 1, 1)                   # [T_HORIZON, n_batch, n_state+n_ctrl, n_state+n_ctrl]

        # 4.2 构造一次性用于整个 batch 的 c
        #     px = -sqrt(goal_weights**2) * x_goal
        #     注意 x_goal_all 是 [N, n_state]，把它与 goal_weights 对应相乘
        px = -torch.sqrt(goal_weights**2).unsqueeze(0) * x_goal_all  # [N, n_state]
        zeros_u = torch.zeros((n_batch, n_ctrl), device=device)      # [N, n_ctrl]
        p_all   = torch.cat((px, zeros_u), dim=1)                    # [N, n_state+n_ctrl]
        #     然后 repeat 到 [T_HORIZON, n_batch, n_state+n_ctrl]
        c = p_all.unsqueeze(0).repeat(T_HORIZON, 1, 1)               # [T_HORIZON, n_batch, n_state+n_ctrl]

        # 4.3 把 x_all 从 [N, n_state] 变成 MPC 要求的 [batch, n_state] 形式（不需要多维扩展）
        #     注意 MPC 的调用通常是 ctrl(x, cost, model)，其中 x 的 shape 应该是 [batch, n_state]
        x_batch = x_all            # [n_batch, n_state]

        # 4.4 构造 QuadCost 对象
        cost = QuadCost(C, c)      # 内部会把 C, c 记录下来，用于梯度传播

        # 4.5 调用一次 MPC，得到 u_opt 的 shape [T_HORIZON, n_batch, n_ctrl]
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
        _, u_opt, _ = ctrl(x_batch, cost, quad)  # u_opt: [T_HORIZON, n_batch, n_ctrl]

        # 4.6 计算误差：只取 u_opt 的第一个时刻 (t=0) 与真实 action 的差
        #     action_all 是 [n_batch, n_ctrl]，u_opt[0] 也是 [n_batch, n_ctrl]
        u_pred = u_opt[0, :, :]               # [n_batch, n_ctrl]
        delta_u = u_pred - action_all         # [n_batch, n_ctrl]

        # 4.7 把 batch 内所有样本的平方误差 sum 掉，得到一个标量 loss
        total_cost = torch.sum(delta_u**2)    # 标量

        # 4.8 反向传播并更新权重
        optimizer.zero_grad()
        total_cost.backward()
        # 可选地做梯度裁剪
        nn_utils.clip_grad_norm_([goal_weights, ctrl_weights], max_norm=1.0)
        optimizer.step()

        # 4.9 打印当前信息
        print(f"Epoch {epoch:03d} | total_cost = {total_cost.item():.6f}")
        # 如果想看梯度大小，也可以：
        print(f"  grad norm goal_weights = {goal_weights.grad.norm().item():.6f}")
        print(f"  grad norm ctrl_weights = {ctrl_weights.grad.norm().item():.6f}")
        print(f"  current goal_weights = {goal_weights.data.cpu().numpy()}")
        print(f"  current ctrl_weights = {ctrl_weights.data.cpu().numpy()}")

def main5():
    # ----------------------------
    # 1. 加载原始数据
    # ----------------------------
    x_arr      = np.load('x_arr.npy')        # 形状 (N, n_state)
    action_arr = np.load('action_arr.npy')   # 形状 (N, n_ctrl)
    x_goal_arr = np.load('x_goal_arr.npy')   # 形状 (N, n_state)
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

if __name__ == "__main__":
    main5()