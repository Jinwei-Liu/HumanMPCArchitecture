import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Personalized_SA.human_model.rlhuman import test
from Personalized_SA.env.quadrotor import *
import torch.optim as optim
import torch.nn.utils as nn_utils
from Personalized_SA.config.config import args

# Load pre-recorded actions and states from a human model
actions, states = test(args, temperature=1, mode=None)
actions = np.array(actions)
states = np.array(states)

x_arr = np.array(states)
action_arr = np.array(actions)
x_goal_arr = x_arr.copy()

print("x_arr.shape:", x_arr.shape)
print("action_arr.shape:", action_arr.shape)
print("x_goal_arr.shape:", x_goal_arr.shape)

DT = 0.01          # Integration step size (s)
T_HORIZON = 15     # MPC prediction steps

quad = Quadrotor_MPC(DT)
n_state = quad.s_dim
n_ctrl = quad.a_dim

n_batch = x_arr.shape[0]     # Use all samples as a batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_all = torch.from_numpy(x_arr).float().to(device)
action_all = torch.from_numpy(action_arr).float().to(device)
x_goal_init = torch.cat((x_all, action_all), dim=1)

# Define column indices that require shared parameters.
# All values within each of these columns will use the same scalar parameter.
shared_columns = [4,5,11,12,13]
total_dims = n_state + n_ctrl

# Create parameters for each column
column_params = {}
for i in range(total_dims):
    if i in shared_columns:
        # For columns with shared parameters, create a single scalar parameter.
        column_params[i] = torch.nn.Parameter(
            torch.tensor(x_goal_init[:, i].mean().item(), device=device),
            requires_grad=True
        )
    else:
        # For other columns, maintain the original per-row parameters.
        column_params[i] = torch.nn.Parameter(x_goal_init[:, i].clone(), requires_grad=True)

def construct_x_goal_param():
    """
    Constructs the complete x_goal_param tensor.
    For specified columns, it uses the same scalar value for all rows.
    """
    x_goal_param = torch.zeros(n_batch, total_dims, device=device)
    
    for col in range(total_dims):
        if col in shared_columns:
            # For shared-parameter columns, broadcast the scalar value to the entire column.
            x_goal_param[:, col] = column_params[col]
        else:
            # For other columns, use independent parameter values for each row.
            x_goal_param[:, col] = column_params[col]
    
    return x_goal_param

goal_weights = torch.ones(n_state, device=device) * 1e-2
goal_weights[0:3] = 0.5       # Set larger initial values for position dimensions.
goal_weights.requires_grad_(True)

ctrl_weights = torch.ones(n_ctrl, device=device) * 1e-1
ctrl_weights.requires_grad_(True)

u_min = torch.tensor([0.0, -20.0, -20.0, -20.0], device=device)
u_max = torch.tensor([100.0,  20.0,  20.0,  20.0], device=device)
u_lower = u_min.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)
u_upper = u_max.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)

optimizer_weights = optim.Adam([goal_weights, ctrl_weights], lr=0.01)
optimizer_goal    = optim.Adam(list(column_params.values()), lr=0.01)

for epoch in range(500):
    # Construct the complete x_goal_param tensor for the current iteration
    x_goal_param = construct_x_goal_param()
    
    q_vector = torch.cat((goal_weights**2, ctrl_weights**2), dim=0)
    Q_diag = torch.diag(q_vector)
    C = Q_diag.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1, 1)

    px = -torch.sqrt(q_vector).unsqueeze(0) * x_goal_param
    p_all = px
    c = p_all.unsqueeze(0).repeat(T_HORIZON, 1, 1)

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
    _, u_opt, _ = ctrl(x_batch, cost, quad)

    u_pred = u_opt[0, :, :]
    delta_u = u_pred - action_all
    total_cost = torch.sum(delta_u**2)

    optimizer_weights.zero_grad()
    optimizer_goal.zero_grad()

    # Alternate between updating weights and goal parameters
    if epoch % 2 == 0:
        # Freeze goal parameters to update weights
        for param in column_params.values():
            param.requires_grad_(False)

        total_cost.backward()
        nn_utils.clip_grad_norm_([goal_weights, ctrl_weights], max_norm=1.0)
        optimizer_weights.step()

        # Unfreeze goal parameters for the next iteration
        for param in column_params.values():
            param.requires_grad_(True)

        print(f"Epoch {epoch:03d} | [Update weights] total_cost = {total_cost.item():.6f}")
        if goal_weights.grad is not None:
            print(f"  grad norm goal_weights = {goal_weights.grad.norm().item():.6f}")
        if ctrl_weights.grad is not None:
            print(f"  grad norm ctrl_weights = {ctrl_weights.grad.norm().item():.6f}")
        print(f"  current goal_weights = {goal_weights.data.cpu().numpy()}")
        print(f"  current ctrl_weights = {ctrl_weights.data.cpu().numpy()}")

    else:
        # Freeze weights to update goal parameters
        goal_weights.requires_grad_(False)
        ctrl_weights.requires_grad_(False)

        total_cost.backward()
        nn_utils.clip_grad_norm_(list(column_params.values()), max_norm=1.0)
        optimizer_goal.step()
        
        # Unfreeze weights for the next iteration
        goal_weights.requires_grad_(True)
        ctrl_weights.requires_grad_(True)

        print(f"Epoch {epoch:03d} | [Update x_goal_arr] total_cost = {total_cost.item():.6f}")
        
        # Print gradients and current values for shared parameter columns
        for col in shared_columns:
            if column_params[col].grad is not None:
                print(f"  grad column {col} = {column_params[col].grad.item():.6f}")
                print(f"  current column {col} value = {column_params[col].data.item():.6f}")
        
        # Print average gradient norm for other (non-shared) columns
        other_columns = [i for i in range(total_dims) if i not in shared_columns]
        other_grad_norms = []
        for col in other_columns:
            if column_params[col].grad is not None:
                other_grad_norms.append(column_params[col].grad.norm().item())
        if other_grad_norms:
            avg_other_grad_norm = np.mean(other_grad_norms)
            print(f"  avg grad norm other columns = {avg_other_grad_norm:.6f}")

print("Training finished!")

# Get the final inferred goal state array
x_goal_final = construct_x_goal_param().data.cpu().numpy()

# Verify if the values within each shared column are identical
print("\n=== Verification of Shared Columns ===")
for col in shared_columns:
    print(f"\nColumn {col}:")
    print(f"  Scalar parameter value: {column_params[col].data.item():.6f}")
    print(f"  First 5 values in column: {x_goal_final[:5, col]}")
    is_uniform = np.allclose(x_goal_final[:, col], x_goal_final[0, col])
    print(f"  Is column {col} uniform? {is_uniform}")
    if is_uniform:
        print(f"  ✓ Column {col} successfully uses a shared parameter.")
    else:
        print(f"  ✗ Column {col} failed to use a shared parameter.")

# Print the final shared parameter values
print(f"\n=== Final Shared Parameter Values ===")
for col in shared_columns:
    print(f"Column {col}: {column_params[col].data.item():.6f}")

last_three_columns = x_goal_final[:, -3:]
row_mean = np.mean(np.sqrt(np.sum(last_three_columns**2, axis=1)))
print(f"\nMean norm of last three columns (angular velocities): {row_mean}")

# Visualization
x_state = x_arr
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_goal_final[:, 0], x_goal_final[:, 1], x_goal_final[:, 2], c='r', marker='o', label='Inferred Goal States')
ax.scatter(x_state[:, 0], x_state[:, 1], x_state[:, 2], c='b', marker='o', label='Initial States')

ax.set_title("Inference of Goal States (3D Scatter Plot)")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.legend()

plt.show()
