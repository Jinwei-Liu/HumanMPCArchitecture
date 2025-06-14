import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from Personalized_SA.human_model.sac import SAC_countinuous
from Personalized_SA.env.quadrotor import *
from typing import Tuple
from Personalized_SA.config.config import args
from Personalized_SA.human_model.rlhuman import scale_to_env
from Personalized_SA.dataset.test_create_data import visualize_path_and_gates
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
            hid_shape=args.hid_shape,
            a_lr=args.actor_lr,
            c_lr=args.critic_lr,
            batch_size=args.batch_size,
            alpha=args.alpha,
            adaptive_alpha=args.adaptive_alpha,
            gamma=args.gamma,
            write=False,
            dvc=self.device
        )
        checkpoint = torch.load(args.load_model, map_location=self.device)
        self.agent.actor.load_state_dict(checkpoint)
        self.agent.actor.to(self.device)

    def select_action(
        self,state: np.ndarray,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        action = self.agent.select_action(state, deterministic=deterministic, temperature=temperature)
        return action

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

    def step(self, state, aim):
        n_batch = 1
        x_all = torch.from_numpy(state).float().to(self.device)
        x_all = x_all.unsqueeze(0)

        self.u_lower = self.u_min.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)
        self.u_upper = self.u_max.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)

        x_goal_param = torch.from_numpy(aim).float().to(self.device)

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
            lqr_iter=2,
            grad_method=GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
            verbose=0
        )
        x, u_opt, _ = ctrl(x_all, cost, self.quad)
        return x.detach().cpu().numpy(), u_opt.detach().cpu().numpy()

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
                lqr_iter=2,
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

            # print(f"Epoch {epoch:03d} | total_cost = {total_cost.item():.6f}")
            # print(f"  grad norm x_goal_param = {x_goal_param.grad.norm().item():.6f}")
            # print(f"  current x_goal_param = {x_goal_param.data.cpu().numpy()}")

        # print("Training complete!")
        return x_goal_param.data.cpu().numpy()
    
class AssistiveMPC:
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

    def run(self, machine_state, human_action, human_goal, human_goal_array, human_action_array):
        machine_state = torch.tensor(machine_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        human_action = torch.tensor(human_action, dtype=torch.float32).to(self.device)
        human_goal = torch.tensor(human_goal, dtype=torch.float32).to(self.device)
        
        n_batch = 1

        q = torch.cat((self.goal_weights, self.ctrl_weights))
        C = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1, 1)

        u_lower = self.u_min.repeat(self.T_HORIZON, 1, 1)   # (T_HORIZON, 4)
        u_upper = self.u_max.repeat(self.T_HORIZON, 1, 1)   # (T_HORIZON, 4)
    
        x_aim =  torch.cat((human_goal, human_action))
        p = -torch.sqrt(q)*x_aim

        c = p.unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1)

        cost = QuadCost(C, c)

        ctrl = mpc.MPC(n_state=self.n_state,
                       n_ctrl=self.n_ctrl,
                       T=self.T_HORIZON,
                       u_lower=u_lower,
                       u_upper=u_upper,
                       lqr_iter=3,
                       grad_method=GradMethods.ANALYTIC,
                       exit_unconverged=False,
                       detach_unconverged=False,
                       verbose=0)
        _, u_opt, _ = ctrl(machine_state, cost, self.quad)
        
        return u_opt[0,0].detach().cpu().numpy()

import collections

def main():
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]
    
    rlhuman = RLHuman(state_dim, action_dim)

    humanmodel = HumanMPC(goal_weights= [1.4150547,   1.0087632,   0.6578214,   0.5467745,   0.74417806,  0.5884964,
                                            0.60344136, -0.01580581,  0.01528102, -0.02966221],
                          ctrl_weights=[0.02876389, 0.04248761, 0.04461192, 0.03424085])
    
    assistvempc = AssistiveMPC(goal_weights=[1,1,1,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5],
                               ctrl_weights=[1,1,1,1],T_HORIZON=15)

    step_idx = 0
    done = False
    obs_dict, _ = env.reset()
    state = obs_dict["human"]
    rewards = 0.0
    state_array=[]
    aim_goal_array=[]

    states = collections.deque([state[:10]] * 100, maxlen=5)  # Adjust length as needed
    actions = collections.deque([np.zeros(action_dim)] * 100, maxlen=5)  # Same length as actions array

    while not done and step_idx < args.max_steps:
        print(f"Step {step_idx}")
        # Sample an action in [-1, 1] and scale to environment bounds
        a_norm = rlhuman.select_action(state, deterministic=False, temperature=1.0)
        env_act = scale_to_env(a_norm, action_low, action_high)

        # Store the current state and action into the stacks (deques)
        states.append(state[:10])
        actions.append(env_act)

        aim_goal = humanmodel.run(np.array(states), np.array(actions))
        # assistvempc_action = assistvempc.run(state[:10], env_act, aim_goal, _, _)
        # print(env_act, assistvempc_action)
        # env_act = assistvempc_action

        if step_idx == 200:
            show_env = Quadrotor_v0(0.01)
            show_env.set_state(state[:10])
            hold_u_x = []
            hold_u_x.append(state[:10])
            store_aim_goal = aim_goal
            for _ in range(15):
                hold_u_x.append(show_env.run(env_act))
            hold_u_x = np.array(hold_u_x)
            x, u =humanmodel.step(state[:10],aim_goal)

        state_array.append(state[:10])
        aim_goal_array.append(aim_goal)

        # Take a step in the environment
        obs_dict, reward, done, info = env.step(env_act)
        next_state = obs_dict["human"] 

        state = next_state
        step_idx += 1
        rewards += reward

    print(f"Total steps: {step_idx}, Total rewards: {rewards}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    state_array = np.array(state_array)
    aim_goal_array = np.array(aim_goal_array)
    
    ax.scatter(aim_goal_array[:, 0], aim_goal_array[:, 1], aim_goal_array[:, 2], c='r', marker='o')
    ax.scatter(state_array[:, 0], state_array[:, 1], state_array[:, 2], c='b', marker='o')
    ax.scatter(x[:,0, 0], x[:,0, 1], x[:,0, 2], c='y', marker='o')
    ax.scatter(hold_u_x[:, 0], hold_u_x[:, 1], hold_u_x[:, 2], c='g', marker='o')
    ax.scatter(store_aim_goal[0], store_aim_goal[1], store_aim_goal[2], c='k', marker='D')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")

    plt.show()
    visualize_path_and_gates(
        start_pos=state_array[0],
        end_pos=state_array[-1],
        gate_positions=env.gate_positions,
        path=aim_goal_array,
        true_path=state_array
    )

if __name__ == "__main__":
    main()
