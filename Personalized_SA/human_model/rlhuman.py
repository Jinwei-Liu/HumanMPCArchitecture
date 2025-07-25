import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from Personalized_SA.human_model.sac import SAC_countinuous
from Personalized_SA.dataset.test_create_data import visualize_path_and_gates
import matplotlib.pyplot as plt

def scale_to_env(a_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Map an action from [-1, 1] to the environment's action bounds [low, high].
    """
    return low + 0.5 * (a_norm + 1.0) * (high - low)

def test_trained_agent(agent: SAC_countinuous,
                       env: QuadrotorRaceEnv,
                       max_steps: int = 2000,
                       temperature: float = 1.0):
    """
    Run one episode in the environment using the given agent.
    Returns the start position and the recorded 3D trajectory.
    """
    obs_dict, _ = env.reset()
    state = obs_dict["human"]
    done = False
    step_idx = 0
    total_reward = 0.0
    trajectory = []
    actions = []
    states = []

    # Record the starting position of the drone
    start_pos = env.quad.get_position().copy()

    while not done and step_idx < max_steps:
        a_norm = agent.select_action(state, deterministic=False, temperature=temperature)
        env_act = scale_to_env(a_norm, env.action_space["low"], env.action_space["high"])
        obs_dict, reward, done, info = env.step(env_act)
        env.render()
        next_state = obs_dict["human"]

        total_reward += reward

        pos = env.quad.get_position().copy()
        trajectory.append(pos)
        actions.append(env_act)
        states.append(state[:10])

        state = next_state
        step_idx += 1

    print(f"== Test episode finished: steps = {step_idx}, done = {done}, info = {info}")
    print(f"== Total reward in this episode = {total_reward:.2f}")
    return start_pos, np.array(trajectory), actions, states

def train(args):
    """
    Train a SAC agent in the QuadrotorRaceEnv and save the actor network.
    """
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]

    device = args.device

    writer = SummaryWriter(log_dir=args.log_dir)

    agent = SAC_countinuous(
        state_dim=state_dim,
        action_dim=action_dim,
        hid_shape=args.hid_shape,
        a_lr=args.actor_lr,
        c_lr=args.critic_lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        adaptive_alpha=args.adaptive_alpha,
        gamma=args.gamma,
        write=True,
        dvc=device
    )

    total_steps = 0
    with tqdm(range(args.episodes), desc="Quadrotor SAC Training") as pbar:
        for ep in pbar:
            obs_dict, _ = env.reset()
            state = obs_dict["human"]
            ep_reward = 0.0
            done = False
            step_idx = 0
            max_steps=args.max_steps

            while not done and step_idx < max_steps:
                # Sample an action in [-1, 1] and scale to environment bounds
                a_norm = agent.select_action(state, deterministic=False)
                env_act = scale_to_env(a_norm, action_low, action_high)

                obs_dict, reward, done, info = env.step(env_act)
                next_state = obs_dict["human"]

                agent.replay_buffer.add(state, a_norm, reward, next_state, done)

                state = next_state
                ep_reward += reward
                total_steps += 1
                step_idx += 1

                if agent.replay_buffer.size > args.batch_size:
                    if step_idx%50 == 1:
                        for _ in range(10):
                            agent.train(writer=writer, total_steps=total_steps)

            pbar.set_postfix(episode_reward=f"{ep_reward:.2f}")
            writer.add_scalar("Reward/episode", ep_reward, ep)
            writer.add_scalar("Reward/total_steps", ep_reward, total_steps)

    # Save the trained actor network
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(agent.actor.state_dict(), args.save_path)
    print(f"Saved trained actor to {args.save_path}")

    env.close()
    writer.close()

def test(args, temperature=1.0, mode='human'):
    """
    Load a trained SAC actor (or use a fresh/random agent) and run a single test episode.
    Visualize the planned vs. actual path.
    """
    env = QuadrotorRaceEnv(dt=0.01, mode=mode)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]

    device = args.device

    if args.load_model:
        print(f"Loading actor weights from {args.load_model} for testing.")
        agent_for_test = SAC_countinuous(
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
            dvc=device
        )
        checkpoint = torch.load(args.load_model, map_location=device)
        agent_for_test.actor.load_state_dict(checkpoint)
    else:
        raise ValueError("Either --load_model must be provided or --use_random must be set for testing.")

    agent_for_test.actor.eval()

    # Run a test episode
    start_pos, true_path, actions, states = test_trained_agent(agent_for_test, env, max_steps=args.max_steps, temperature=temperature)

    n_state = 10
    states = np.array(states)
    fig, axes = plt.subplots(n_state,      # 行数 = 状态维度
                        1,            # 一列
                        sharex=True,  # 共用 x 轴
                        figsize=(6, 1.8*n_state)) 

    for dim in range(n_state):
        ax = axes[dim]                    # 当前子图
        ax.plot(states[:, dim])            # 画出 dim 维
        ax.set_ylabel(f"x[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Timestep")       # 只在最后一行标注
    fig.suptitle("Evolution of state vector x", y=1.02)
    fig.tight_layout()
    plt.show()

    gate_positions = np.array(env.gate_positions)
    end_pos = true_path[-1]

    # Build a simple planned path: start → each gate center → end
    planned_points = [start_pos] + [gp.copy() for gp in gate_positions] + [end_pos]
    planned_path = np.vstack(planned_points)

    env.close()

    visualize_path_and_gates(
        start_pos=start_pos,
        end_pos=end_pos,
        gate_positions=gate_positions,
        path=planned_path,
        true_path=true_path
    )

    return actions, states

from Personalized_SA.config.config import args

if __name__ == "__main__":
    # train(args)
    test(args, temperature=1)