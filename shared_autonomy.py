import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from Personalized_SA.human_model.sac import SAC_countinuous
from Personalized_SA.env.quadrotor import *
from typing import Tuple
from Personalized_SA.config.config import args

class RLHuman:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        self.device = args.device

        self.agent = SAC_countinuous(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=args.hidden_sizes,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            batch_size=args.batch_size,
            alpha=args.alpha,
            adaptive_alpha=args.adaptive_alpha,
            gamma=args.gamma,
            write=False,
            device=self.device,
        )

        checkpoint = torch.load(args.load_model, map_location=self.device)
        self.agent.actor.load_state_dict(checkpoint)
        self.agent.actor.to(self.device)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)

        with torch.no_grad():
            action_tensor = self.agent.select_action(
                state_tensor, deterministic=deterministic, temperature=temperature
            )

        return action_tensor.cpu().numpy().flatten()

class HumanMPC:
    def __init__(self, goal_weights, ctrl_weights, DT=0.01, T_HORIZON=15):
        self.DT = DT
        self.T_HORIZON = T_HORIZON
        self.quad = Quadrotor_MPC(self.DT)
        self.n_state, self.n_ctrl = self.quad.s_dim, self.quad.a_dim
        self.device = args.device
        self.goal_weights = torch.Tensor(goal_weights).to(self.device)
        self.ctrl_weights = torch.Tensor(ctrl_weights).to(self.device)

        self.u_min = torch.tensor([0.0, -20.0, -20.0, -20.0], device=self.device)
        self.u_max = torch.tensor([100.0, 20.0, 20.0, 20.0], device=self.device)

    def run(self, states, actions):
        x_arr = states
        action_arr = actions
        n_batch = x_arr.shape[0]
        x_all = torch.from_numpy(x_arr).float().to(self.device)
        action_all = torch.from_numpy(action_arr).float().to(self.device)

        # 根据 n_batch 动态重置 u_lower、u_upper
        self.u_lower = self.u_min.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)
        self.u_upper = self.u_max.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)

        # 后面的初始化 x_goal_param、优化循环同前面示例
        x_goal_init = torch.from_numpy(x_arr[0]).float().to(self.device)
        x_goal_param = torch.nn.Parameter(x_goal_init.clone(), requires_grad=True)
        optimizer_goal = optim.Adam([x_goal_param], lr=0.001)

        for epoch in range(10):
            # 构造 C、c，调用 MPC 时传入 self.u_lower/self.u_upper
            q_vector = torch.cat((self.goal_weights**2, self.ctrl_weights**2), dim=0)
            Q_diag = torch.diag(q_vector)
            C = Q_diag.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1, 1)

            x_goal = x_goal_param.unsqueeze(0).repeat(n_batch, 1)
            px = -torch.sqrt(self.goal_weights**2).unsqueeze(0) * x_goal
            zeros_u = torch.zeros((n_batch, self.n_ctrl), device=self.device)
            p_all = torch.cat((px, zeros_u), dim=1)
            c = p_all.unsqueeze(0).repeat(self.T_HORIZON, 1, 1)
            cost = QuadCost(C, c)

            ctrl = mpc.MPC(
                n_state=self.n_state,
                n_ctrl=self.n_ctrl,
                T=self.T_HORIZON,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                lqr_iter=5,
                grad_method=GradMethods.ANALYTIC,
                exit_unconverged=False,
                detach_unconverged=False,
                verbose=0
            )
            _, u_opt, _ = ctrl(x_all, cost, self.quad)

            u_pred = u_opt[0, :, :]
            delta_u = u_pred - action_all
            total_cost = torch.sum(delta_u**2)
            optimizer_goal.zero_grad()
            total_cost.backward()
            nn_utils.clip_grad_norm_([x_goal_param], max_norm=1.0)
            optimizer_goal.step()

            print(f"Epoch {epoch:03d} | total_cost = {total_cost.item():.6f}")
            print(f"  grad norm x_goal_param = {x_goal_param.grad.norm().item():.6f}")
            print(f"  current x_goal_param = {x_goal_param.data.cpu().numpy()}")

        print("Training complete!")
        return x_goal_param.data.cpu().numpy()

from Personalized_SA.human_model.rlhuman import test
def test_human_mpc():
    # Load the action and state data from your test set
    actions, states = test(args, temperature=1)
    actions = np.array(actions)
    states = np.array(states)

    # Print the shapes of the states and actions for debugging
    print("states.shape:", states.shape)  # (N, n_state)
    print("actions.shape:", actions.shape)  # (N, n_ctrl)

    # Goal and control weights (can be customized based on the problem)
    goal_weights = [1.0, 1.0, 1.0, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]  # Example goal weights
    ctrl_weights = [1e-5, 1e-5, 1e-5, 1e-5]  # Example control weights

    # Initialize the HumanMPC class with the weights
    mpc = HumanMPC(goal_weights=goal_weights, ctrl_weights=ctrl_weights)

    # Test the `run` method to optimize x_goal
    print("Running optimization for x_goal...")
    x_goal_final = mpc.run(states[:5,:], actions[:5,:])  # Run the optimization process

    # Print the final x_goal after optimization
    print(f"Final x_goal: {x_goal_final}")

    # Optionally, visualize the result
    x_state = states  # Use the state data to visualize the result
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_goal_final[:, 0], x_goal_final[:, 1], x_goal_final[:, 2], c='r', marker='o', label="Optimized x_goal")
    ax.scatter(x_state[:, 0], x_state[:, 1], x_state[:, 2], c='b', marker='o', label="Actual State")

    ax.set_title("Inference of x_goal during training (3D Scatter Plot)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.legend()

    plt.show()

# Run the test
test_human_mpc()
