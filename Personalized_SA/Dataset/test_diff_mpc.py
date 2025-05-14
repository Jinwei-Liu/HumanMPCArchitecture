import sys
import torch.optim as optim
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from Personalized_SA.env.quadrotor import *
from torchviz import make_dot

def main():
    x_arr = np.load('x_arr.npy')
    action_arr = np.load('action_arr.npy')
    print(x_arr.shape)
    print(action_arr.shape)

    DT          = 0.01          # 积分步长  (s)
    T_HORIZON   = 15         # MPC 预测步数

    quad = Quadrotor_MPC(DT)

    n_state, n_ctrl   = quad.s_dim, quad.a_dim

    w_pos, w_vel      = 1., 0.001
    w_quat            = 0.001
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
    u_max = torch.tensor([100.0,  20.0,  20.0,  20.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)   # (25, 4)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)   # (25, 4)

    steps = len(x_arr)

    x = torch.from_numpy(x_arr[0]).float()
    x = x.unsqueeze(0)
    action = torch.from_numpy(action_arr[1]).float()
    action = action.unsqueeze(0)

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
        lqr_iter=10,
        grad_method=GradMethods.AUTO_DIFF,
        exit_unconverged = False,
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

        # Optionally, you can also print the updated optimal control input and action
        print("Updated u_opt:", u_opt[0])
        print("Updated action:", action)




if __name__ == "__main__":
    main()