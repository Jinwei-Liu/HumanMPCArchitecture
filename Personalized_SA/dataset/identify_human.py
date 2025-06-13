import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Personalized_SA.human_model.rlhuman import test
from Personalized_SA.env.quadrotor import *
import torch.optim as optim
import torch.nn.utils as nn_utils
from Personalized_SA.config.config import args

actions, states = test(args, temperature=1)
actions = np.array(actions)
states = np.array(states)

x_arr      = np.array(states)
action_arr = np.array(actions)
x_goal_arr = x_arr.copy() 

print("x_arr.shape:", x_arr.shape)         # (N, n_state)
print("action_arr.shape:", action_arr.shape) # (N, n_ctrl)
print("x_goal_arr.shape:", x_goal_arr.shape) # (N, n_state)

DT        = 0.01          # Integration step size (s)
T_HORIZON = 15            # MPC prediction steps

quad      = Quadrotor_MPC(DT)
n_state   = quad.s_dim
n_ctrl    = quad.a_dim

n_batch = x_arr.shape[0]     # Use all samples as a batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_all      = torch.from_numpy(x_arr).float().to(device)      # [N, n_state]
action_all = torch.from_numpy(action_arr).float().to(device) # [N, n_ctrl]
x_goal_init = torch.from_numpy(x_goal_arr).float().to(device) # [N, n_state]

x_goal_param = torch.nn.Parameter(x_goal_init.clone(), requires_grad=True)  # [N, n_state]

goal_weights = torch.ones(n_state, device=device) * 1e-2
goal_weights[0:3] = 0.5       # Set larger initial values for the first 3 dimensions: posture/position
goal_weights.requires_grad_(True)

ctrl_weights = torch.ones(n_ctrl, device=device) * 1e-2
ctrl_weights.requires_grad_(True)

u_min = torch.tensor([0.0, -20.0, -20.0, -20.0], device=device)
u_max = torch.tensor([100.0,  20.0,  20.0,  20.0], device=device)
u_lower = u_min.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)  # [T_HORIZON, N, n_ctrl]
u_upper = u_max.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)  # [T_HORIZON, N, n_ctrl]

optimizer_weights = optim.Adam([goal_weights, ctrl_weights], lr=0.01)
optimizer_goal    = optim.Adam([x_goal_param], lr=0.01)

for epoch in range(500):
    q_vector = torch.cat((goal_weights**2, ctrl_weights**2), dim=0)  # [n_state + n_ctrl]
    Q_diag = torch.diag(q_vector)                           # [n_state+n_ctrl, n_state+n_ctrl]
    C = Q_diag.unsqueeze(0).unsqueeze(0)                     # [1, 1, n_state+n_ctrl, n_state+n_ctrl]
    C = C.repeat(T_HORIZON, n_batch, 1, 1)                   # [T_HORIZON, N, n_state+n_ctrl, n_state+n_ctrl]

    px = -torch.sqrt(goal_weights**2).unsqueeze(0) * x_goal_param  # [N, n_state]
    zeros_u = torch.zeros((n_batch, n_ctrl), device=device)         # [N, n_ctrl]
    p_all   = torch.cat((px, zeros_u), dim=1)                       # [N, n_state+n_ctrl]
    c = p_all.unsqueeze(0).repeat(T_HORIZON, 1, 1)                 # [T_HORIZON, N, n_state+n_ctrl]

    x_batch = x_all

    cost = QuadCost(C, c)

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

    u_pred = u_opt[0, :, :]               # [N, n_ctrl]
    delta_u = u_pred - action_all         # [N, n_ctrl]
    total_cost = torch.sum(delta_u**2)    # Scalar

    optimizer_weights.zero_grad()
    optimizer_goal.zero_grad()

    if epoch % 2 == 0:
        x_goal_param.requires_grad_(False)

        total_cost.backward()  # Compute gradient with respect to goal_weights/ctrl_weights
        nn_utils.clip_grad_norm_([goal_weights, ctrl_weights], max_norm=1.0)
        optimizer_weights.step()  # Update goal_weights, ctrl_weights

        x_goal_param.requires_grad_(True)

        print(f"Epoch {epoch:03d} | [Update weights] total_cost = {total_cost.item():.6f}")
        print(f"  grad norm goal_weights = {goal_weights.grad.norm().item():.6f}")
        print(f"  grad norm ctrl_weights = {ctrl_weights.grad.norm().item():.6f}")
        print(f"  current goal_weights = {goal_weights.data.cpu().numpy()}")
        print(f"  current ctrl_weights = {ctrl_weights.data.cpu().numpy()}")

    else:
        goal_weights.requires_grad_(False)
        ctrl_weights.requires_grad_(False)

        total_cost.backward()  # Compute gradient with respect to x_goal_param
        nn_utils.clip_grad_norm_([x_goal_param], max_norm=1.0)
        optimizer_goal.step()  # Update x_goal_param
        
        goal_weights.requires_grad_(True)
        ctrl_weights.requires_grad_(True)

        print(f"Epoch {epoch:03d} | [Update x_goal_arr] total_cost = {total_cost.item():.6f}")
        print(f"  grad norm x_goal_param = {x_goal_param.grad.norm().item():.6f}")
        print(f"  current x_goal_param[:5] = {x_goal_param.data.cpu().numpy()[:5]}")

print("Training finished!")

x_goal_final = x_goal_param.data.cpu().numpy()  # Get the final inferred goal state array
x_state = x_arr
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_goal_final[:, 0], x_goal_final[:, 1], x_goal_final[:, 2], c='r', marker='o')
ax.scatter(x_state[:, 0], x_state[:, 1], x_state[:, 2], c='b', marker='o')

ax.set_title("Inference of x_goal during training (3D Scatter Plot)")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

plt.show()
