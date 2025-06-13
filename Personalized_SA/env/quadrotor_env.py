import numpy as np
import sys
import os
import matplotlib.pyplot as plt  
from typing import Optional

try:
    import pybullet as p
except ImportError:
    p = None
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Personalized_SA.env.quadrotor import Quadrotor_v0

class QuadrotorRaceEnv:
    def __init__(self, dt: float = 0.01, mode: str = 'none'):
        """
        mode: 'human' for visualized environment (pybullet GUI),
              any other value skips rendering setup.
        """
        self.quad = Quadrotor_v0(dt)
        self.dt = dt
        self.mode = mode

        # Action space bounds for sampling
        self.action_space = {
            'low':  np.array([0.0,  -20.0, -20.0, -20.0], dtype=np.float32),
            'high': np.array([20.0,  20.0,  20.0,  20.0], dtype=np.float32)
        }

        self.observation_dim_machine = self.quad.s_dim
        self.observation_dim_human = self.quad.s_dim + 3  # 3 for gate position (x, y, z)

        # Gate trajectory parameters
        self.num_gates = 4
        self.gate_radius = 0.5
        self.current_gate_idx = 0
        self.gate_positions_raw = np.array([
            [2.0, 5.0, 2.0],
            [-2.0, 10.0, 2.0],
            [2.0, 15.0, 2.0],
            [-2.0, 20.0, 2.0]
        ], dtype=np.float32)
        self.noise_bounds = np.array([
            [-0.5, 0.5],  
            [-1.0, 1.0], 
            [-3.0, 3.0] 
        ])

        # Visualization handles
        self.gate_ids = []

        # Initialize PyBullet for rendering if in human mode
        if self.mode == 'human':
            if p is None:
                raise ImportError("pybullet is required for human mode but not installed.")
            p.connect(p.GUI, options="--opengl2")
            p.setAdditionalSearchPath(
                os.path.join(os.path.dirname(__file__), 'util'),
            )
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(self.dt)
            self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -5])
            self.quadrotor_id = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, 0])
        else:
            # Skip visualization setup
            self.plane_id = None
            self.quadrotor_id = None

        # Safety bounds
        self.pos_bounds = np.array([-5.0, 20.0], dtype=np.float32)
        self.vel_bounds = np.array([-100.0, 100.0], dtype=np.float32)

        # Reward coefficients
        self.gate_pass_reward   = 10.0
        self.dist_penalty_scale = 10.0
        self.vel_penalty_scale  = -1
        self.threshold_vel = 3
        self.crash_penalty      = -0.0
        self.time_penalty       = -0.0
        self.history_reward_gate = None

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        # Sample new gate positions
        self.gate_positions = self.gate_positions_raw + np.random.uniform(
            low=self.noise_bounds[:, 0],
            high=self.noise_bounds[:, 1],
            size=self.gate_positions_raw.shape
        )
        state = self.quad.reset()
        self.current_gate_idx = 0
        self.history_reward_gate = None

        # Clear existing debug lines
        if self.mode == 'human' and self.gate_ids:
            for gid in self.gate_ids:
                p.removeBody(gid)
            self.gate_ids = []

        # Draw gates once per reset
        if self.mode == 'human':
            for gate_pos in self.gate_positions:
                gid = p.loadURDF("gate.urdf",
                    basePosition=gate_pos.tolist(),
                    useFixedBase=True
                )
                self.gate_ids.append(gid)

        obs_machine = state.astype(np.float32)
        current_gate = self.gate_positions[self.current_gate_idx]
        obs_human = np.concatenate([state, current_gate]).astype(np.float32)

        return {'machine': obs_machine, 'human': obs_human}, {}

    def step(self, action: np.ndarray):
        state = self.quad.run(action)
        pos  = self.quad.get_position()
        vel  = self.quad.get_velocity()
        quat = self.quad.get_quaternion()

        done = False
        info = {}

        # Check out-of-bounds
        if np.any(pos < self.pos_bounds[0]) or np.any(pos > self.pos_bounds[1]):
            done = True
            info['termination'] = 'out_of_bounds'

        # Check excessive velocity
        if np.any(vel < self.vel_bounds[0]) or np.any(vel > self.vel_bounds[1]):
            done = True
            info['termination'] = 'excessive_velocity'

        # Check quaternion validity
        if np.abs(np.linalg.norm(quat) - 1.0) > 0.1:
            done = True
            info['termination'] = 'invalid_quaternion'

        reward = self._compute_reward(pos, vel)

        current_gate = self.gate_positions[self.current_gate_idx]
        dist_to_gate = np.linalg.norm(pos - current_gate)

        if dist_to_gate < self.gate_radius:
            reward += self.gate_pass_reward
            info['gate_passed'] = self.current_gate_idx
            self.current_gate_idx += 1
            self.history_reward_gate = None

            if self.current_gate_idx >= self.num_gates:
                done = True
                info['termination'] = 'all_gates_passed'

        obs_machine = state.astype(np.float32)
        next_gate = (
            self.gate_positions[self.current_gate_idx]
            if self.current_gate_idx < self.num_gates
            else self.gate_positions[-1]
        )
        obs_human = np.concatenate([state, next_gate]).astype(np.float32)

        return {'machine': obs_machine, 'human': obs_human}, reward, done, info

    def _compute_reward(self, pos: np.ndarray, vel: np.ndarray) -> float:
        reward = self.time_penalty
        current_gate = self.gate_positions[self.current_gate_idx]
        dist_to_gate = np.linalg.norm(pos - current_gate)
        reward_gate = - self.dist_penalty_scale * dist_to_gate
        if self.history_reward_gate is None:
            self.history_reward_gate = reward_gate

        reward += reward_gate - self.history_reward_gate
        self.history_reward_gate = reward_gate
        # reward += self.vel_penalty_scale * np.linalg.norm(vel)
        if np.linalg.norm(vel) > self.threshold_vel:
            reward += self.vel_penalty_scale

        if (
            np.any(pos < self.pos_bounds[0]) or
            np.any(pos > self.pos_bounds[1]) or
            np.any(vel < self.vel_bounds[0]) or
            np.any(vel > self.vel_bounds[1])
        ):
            reward += self.crash_penalty

        return float(reward)

    def render(self, mode: str = 'human'):
        if self.mode != 'human':
            return
        pos_model = self.quad.get_position()
        quat_model = self.quad.get_quaternion()
        pb_quat = [quat_model[1], quat_model[2], quat_model[3], quat_model[0]]
        p.resetBasePositionAndOrientation(
            self.quadrotor_id,
            pos_model.tolist(),
            pb_quat
        )
        # Only update camera; gates already drawn
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=pos_model.tolist()
        )
        time.sleep(0.01)

    def close(self):
        """
        Clean up and close the environment.
        """
        if self.mode == 'human' and p is not None:
            p.disconnect()

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(
            low=self.action_space['low'],
            high=self.action_space['high'],
            size=self.action_space['low'].shape
        ).astype(np.float32)


def visualize_gates(gate_positions: np.ndarray, gate_radius: float = 0.5):
    """
    Visualize gate positions in 3D and draw a true-radius circle for each gate.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw gate centers
    ax.scatter(
        gate_positions[:, 0],
        gate_positions[:, 1],
        gate_positions[:, 2],
        c='red',
        marker='o',
        s=30,
        label='Gate centers'
    )

    # Parameters for drawing circles
    theta = np.linspace(0, 2 * np.pi, 100)

    for i, (x0, y0, z0) in enumerate(gate_positions):
        x_circle = x0 + gate_radius * np.cos(theta)
        y_circle = np.full_like(theta, y0)
        z_circle = z0 + gate_radius * np.sin(theta)

        ax.plot(x_circle, y_circle, z_circle, c='blue', linewidth=1.5)
        ax.text(x0, y0, z0 + 0.1, f'Gate {i+1}', color='black')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Gate Positions with True Radius')

    max_range = np.array([
        gate_positions[:, 0].max() - gate_positions[:, 0].min(),
        gate_positions[:, 1].max() - gate_positions[:, 1].min(),
        gate_positions[:, 2].max() - gate_positions[:, 2].min()
    ]).max()

    mid_x = (gate_positions[:, 0].max() + gate_positions[:, 0].min()) / 2
    mid_y = (gate_positions[:, 1].max() + gate_positions[:, 1].min()) / 2
    mid_z = (gate_positions[:, 2].max() + gate_positions[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    env = QuadrotorRaceEnv(dt=0.01, mode='human')
    obs, _ = env.reset(seed=42)
    print("Test Reset output:", obs)

    # Visualize gates only
    visualize_gates(env.gate_positions)

    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        action = env.sample_action()
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        step_count += 1
        print(f"Step {step_count}: Reward = {reward:.2f}, Done = {done}, Info = {info}")

    print(f"Total reward: {total_reward:.2f}")
    env.close()
