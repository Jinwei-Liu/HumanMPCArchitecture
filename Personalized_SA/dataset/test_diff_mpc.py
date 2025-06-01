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
        

if __name__ == "__main__":
    main3()