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
from Personalized_SA.visualization.visualization import plot_state_3d
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
        aim[4]=-0.009359
        aim[5]=-0.010854
        aim[11]=-0.086476
        aim[12]=-0.072684
        aim[13]=-0.014102

        n_batch = 1
        x_all = torch.from_numpy(state).float().to(self.device)
        x_all = x_all.unsqueeze(0)

        self.u_lower = self.u_min.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)
        self.u_upper = self.u_max.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)

        x_goal_param = torch.from_numpy(aim).float().to(self.device)

        q_vector = torch.cat((self.goal_weights**2, self.ctrl_weights**2), dim=0)
        Q_diag = torch.diag(q_vector)
        C = Q_diag.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1, 1)

        x_goal = x_goal_param.unsqueeze(0).repeat(n_batch, 1)
        # px = -torch.sqrt(self.goal_weights**2).unsqueeze(0) * x_goal
        # zeros_u = torch.zeros((n_batch, self.n_ctrl), device=self.device)
        # p_all = torch.cat((px, zeros_u), dim=1)
        p_all = -torch.sqrt(q_vector).unsqueeze(0) * x_goal
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
            backprop=False,
            verbose=0
        )
        x, u_opt, _ = ctrl(x_all, cost, self.quad)
        return x.detach().cpu().numpy(), u_opt.detach().cpu().numpy()

    def run(self, states, actions):
        n_batch = states.shape[0]
        x_all = torch.from_numpy(states).float().to(self.device)
        action_all = torch.from_numpy(actions).float().to(self.device)

        self.u_lower = self.u_min.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)
        self.u_upper = self.u_max.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1).to(self.device)

        x_goal_init = torch.from_numpy(np.concatenate((states[0],actions[0]),axis=0)).float().to(self.device)
        x_goal_param = torch.nn.Parameter(x_goal_init.clone(), requires_grad=True)
        optimizer_goal = optim.Adam([x_goal_param], lr=0.01)

        for epoch in range(3):
            q_vector = torch.cat((self.goal_weights**2, self.ctrl_weights**2), dim=0)
            Q_diag = torch.diag(q_vector)
            C = Q_diag.unsqueeze(0).unsqueeze(0).repeat(self.T_HORIZON, n_batch, 1, 1)

            x_goal = x_goal_param.unsqueeze(0).repeat(n_batch, 1)
            # px = -torch.sqrt(self.goal_weights**2).unsqueeze(0) * x_goal
            # zeros_u = torch.zeros((n_batch, self.n_ctrl), device=self.device)
            # p_all = torch.cat((px, zeros_u), dim=1)
            p_all = -torch.sqrt(q_vector).unsqueeze(0) * x_goal
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

    def run(self, machine_state, human_actions, human_states):
        machine_state = torch.tensor(machine_state, dtype=torch.float32).to(self.device).unsqueeze(0)
        human_action = torch.tensor(human_actions[0], dtype=torch.float32).to(self.device)
        human_goal = torch.tensor(human_states[0], dtype=torch.float32).to(self.device)
        
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

def cal_error(true_states, predict_states, predict_steps):
    true_steps = true_states.shape[0]
    errors_list = []

    for dim in range(true_states.shape[1]):
        errors = []
        for step in range(true_steps - predict_steps):  
            true_values = true_states[step:step + predict_steps, dim]
            predict_values = predict_states[step, :predict_steps, dim]
            error = np.abs(true_values - predict_values)
            errors.append(np.mean(error))
        errors_list.append(np.mean(errors))
    return errors_list

import collections

def main():
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]

    rlhuman = RLHuman(state_dim, action_dim)

    humanmodel = HumanMPC(goal_weights= args.goal_weights,
                          ctrl_weights= args.ctrl_weights, T_HORIZON=50)

    step_idx = 0
    done = False
    obs_dict, _ = env.reset()
    state = obs_dict["human"]
    rewards = 0.0
    state_array=[]
    aim_goal_array=[]
    store_predict_array=[]
    hold_u_x_array=[]

    states = collections.deque([state[:10]] * 100, maxlen=3)  # Adjust length as needed
    actions = collections.deque([np.zeros(action_dim)] * 100, maxlen=3)  # Same length as actions array

    while not done and step_idx < args.max_steps:
        print(f"Step {step_idx}")
        # Sample an action in [-1, 1] and scale to environment bounds
        a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
        env_act = scale_to_env(a_norm, action_low, action_high)

        # Store the current state and action into the stacks (deques) v 
        states.append(state[:10])
        actions.append(env_act)

        aim_goal = humanmodel.run(np.array(states), np.array(actions))

        show_env = Quadrotor_v0(0.01)
        show_env.set_state(state[:10])
        hold_u_x = []
        hold_u_x.append(state[:10])
        for _ in range(50):
            hold_u_x.append(show_env.run(env_act))
        hold_u_x = np.array(hold_u_x)
        hold_u_x_array.append(hold_u_x)

        x, u =humanmodel.step(state[:10],aim_goal)
        store_predict_array.append(np.squeeze(x))

        state_array.append(state[:10])
        aim_goal_array.append(aim_goal)

        # Take a step in the environment
        obs_dict, reward, done, info = env.step(env_act)
        next_state = obs_dict["human"] 

        state = next_state
        step_idx += 1
        rewards += reward

    print(f"Total steps: {step_idx}, Total rewards: {rewards}")

    state_array = np.array(state_array)
    aim_goal_array = np.array(aim_goal_array)
    store_predict_array = np.array(store_predict_array)
    hold_u_x_array = np.array(hold_u_x_array)
    gate_positions = np.array(env.gate_positions)
    np.savez_compressed(
        args.visualization_save_path,
        state_array=state_array,
        aim_goal_array=aim_goal_array,
        store_predict_array=store_predict_array,
        hold_u_x_array=hold_u_x_array,
        gate_positions=gate_positions
    )

    plot_state_3d(state_array,
                store_predict_array,
                hold_u_x_array,
                gate_positions,
                save_path="Prediction_error.pdf")

    print("5 steps:")
    print(cal_error(state_array,store_predict_array,5))
    print(cal_error(state_array,hold_u_x_array,5))

    print("10 steps:")
    print(cal_error(state_array,store_predict_array,10))
    print(cal_error(state_array,hold_u_x_array,10))

    print("20 steps:")
    print(cal_error(state_array,store_predict_array,20))
    print(cal_error(state_array,hold_u_x_array,20))
    
    print("30 steps:")
    print(cal_error(state_array,store_predict_array,30))
    print(cal_error(state_array,hold_u_x_array,30))
    
    print("40 steps:")
    print(cal_error(state_array,store_predict_array,40))
    print(cal_error(state_array,hold_u_x_array,40))

    print("50 steps:")
    print(cal_error(state_array,store_predict_array,50))
    print(cal_error(state_array,hold_u_x_array,50))

    visualize_path_and_gates(
        start_pos=state_array[0],
        end_pos=state_array[-1],
        gate_positions=env.gate_positions,
        path=aim_goal_array,
        true_path=state_array
    )

def draw_results():
    data = np.load(args.visualization_save_path)
    state_array = data['state_array']
    aim_goal_array = data['aim_goal_array']
    store_predict_array = data['store_predict_array']
    hold_u_x_array = data['hold_u_x_array']
    gate_positions = data['gate_positions']

    plot_state_3d(state_array,
            store_predict_array,
            hold_u_x_array,
            gate_positions)

if __name__ == "__main__":
    draw_results()