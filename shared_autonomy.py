import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from Personalized_SA.config.config import args

# 状态索引常量（根据你的四旋翼模型）
kPosX, kPosY, kPosZ = 0, 1, 2
kQuatW, kQuatX, kQuatY, kQuatZ = 3, 4, 5, 6
kVelX, kVelY, kVelZ = 7, 8, 9

class Quadrotor_MPC_CasADi:
    """CasADi版本的四旋翼动力学模型"""
    def __init__(self, dt):
        self.s_dim = 10
        self.a_dim = 4
        self._gz = 9.81
        self._dt = dt

class AssistiveMPC:
    def __init__(self, goal_weights, ctrl_weights, obstacles=None, cbf_gamma=1.0, DT=0.01, T_HORIZON=15):
        self.DT = DT
        self.T_HORIZON = T_HORIZON
        self.quad = Quadrotor_MPC_CasADi(self.DT)
        self.n_state, self.n_ctrl = self.quad.s_dim, self.quad.a_dim
        
        # CBF参数
        self.cbf_gamma = cbf_gamma
        self.obstacles = obstacles if obstacles is not None else []
        
        # 权重参数
        self.goal_weights = np.array(goal_weights, dtype=np.float64)
        self.ctrl_weights = np.array(ctrl_weights, dtype=np.float64)
        
        # 控制约束
        self.u_min = np.array([0.0, -20.0, -20.0, -20.0], dtype=np.float64)
        self.u_max = np.array([100.0, 20.0, 20.0, 20.0], dtype=np.float64)
        
        print(f"AssistiveMPC initialized with T_HORIZON={self.T_HORIZON}")
        print(f"Obstacles: {len(self.obstacles)} defined")
        
        # 设置CasADi优化问题
        self._setup_optimizer()
    
    def add_obstacle(self, x_obs, y_obs, z_obs, radius):
        """添加球形障碍物"""
        obstacle = {
            'x': float(x_obs),
            'y': float(y_obs), 
            'z': float(z_obs),
            'radius': float(radius)
        }
        self.obstacles.append(obstacle)
        print(f"Added obstacle at ({x_obs}, {y_obs}, {z_obs}) with radius {radius}")
        # 重新设置优化器以包含新的障碍物约束
        self._setup_optimizer()
    
    def _barrier_function(self, x, obstacle):
        """计算CBF函数值 h(x) = dist^2 - radius^2"""
        pos_x = x[kPosX]
        pos_y = x[kPosY] 
        pos_z = x[kPosZ]
        
        dist_sq = (pos_x - obstacle['x'])**2 + (pos_y - obstacle['y'])**2 + (pos_z - obstacle['z'])**2
        h = dist_sq - obstacle['radius']**2
        return h
    
    def _setup_optimizer(self):
        """设置CasADi优化器 - 按照图片中的MPC-CBF公式"""
        # 创建优化变量
        X = ca.SX.sym('X', self.n_state, self.T_HORIZON + 1)  # 状态轨迹
        U = ca.SX.sym('U', self.n_ctrl, self.T_HORIZON)       # 控制轨迹
        
        # 参数: [初始状态, 人类控制输入序列, 人类状态序列]
        # 人类控制输入: T_HORIZON x n_ctrl
        # 人类状态: T_HORIZON x n_state  
        param_size = self.n_state + self.T_HORIZON * self.n_ctrl + self.T_HORIZON * self.n_state
        P = ca.SX.sym('P', param_size)
        
        print(f"Parameter vector size: {param_size}")
        print(f"  - Initial state: {self.n_state}")
        print(f"  - Human control inputs: {self.T_HORIZON * self.n_ctrl}")
        print(f"  - Human target states: {self.T_HORIZON * self.n_state}")
        
        # 创建目标函数 - 根据图片中的公式
        obj = 0
        g = []  # 约束
        
        # 初始条件约束
        g.append(X[:, 0] - P[:self.n_state])
        
        # 解析参数
        idx = self.n_state
        # 人类控制输入序列 u^h_t+k
        U_human = ca.reshape(P[idx:idx + self.T_HORIZON * self.n_ctrl], 
                            self.n_ctrl, self.T_HORIZON)
        idx += self.T_HORIZON * self.n_ctrl
        
        # 人类目标状态序列 x^h_t+k  
        X_human = ca.reshape(P[idx:idx + self.T_HORIZON * self.n_state],
                            self.n_state, self.T_HORIZON)
        
        # 构建目标函数 - 按照图片公式: ||u_t - u^h_t||² + Σ||x_t+k - x^h_t+k||² + ||u_t+k||²
        # 1. 人类控制跟踪项: ||u_t+k - u^h_t+k||²_R_m1
        u_error = U[:, 0] - U_human[:, 0]
        Q_u_track = ca.diag(self.ctrl_weights)
        obj += ca.mtimes([u_error.T, Q_u_track, u_error])

        for k in range(self.T_HORIZON):
            # 2. 状态跟踪项: ||x_t+k - x^h_t+k||²_Q_m
            x_error = X[:, k] - X_human[:, k]  
            Q_x = ca.diag(self.goal_weights)
            obj += ca.mtimes([x_error.T, Q_x, x_error])
            
            # 3. 控制努力项: ||u_t+k||²_R_m2 (控制输入正则化)
            Q_u_reg = ca.diag(self.ctrl_weights * 0.1)  # 较小权重
            obj += ca.mtimes([U[:, k].T, Q_u_reg, U[:, k]])
            
            # 动力学约束: x_t+k+1 = f(x_t+k, u_t+k)
            x_next = self._quadrotor_dynamics(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
        
        for k in range(self.T_HORIZON):  # 包括终端状态
            for obs in self.obstacles:
                h_curr = self._barrier_function(X[:, k], obs)
                h_next = self._barrier_function(X[:, k+1], obs)
                
                # CBF约束: h_next - h_curr >= -gamma * h_curr
                # 重写为: h_next >= (1 - gamma) * h_curr
                cbf_constraint = h_next - (1 - self.cbf_gamma) * h_curr
                g.append(cbf_constraint)
        
        # 将变量展平为向量
        opt_variables = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )
        
        # 设置优化问题
        nlp_prob = {
            'f': obj,
            'x': opt_variables,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        # 设置求解器选项
        opts = {
            'ipopt': {
                'max_iter': 100,
                'print_level': 0,
                'acceptable_tol': 1e-4,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }
        
        # 创建求解器
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        # 设置变量界限
        self._setup_bounds()

    def _setup_bounds(self):
        """设置变量界限"""
        # 状态变量界限
        self.lbx_X = -ca.inf * ca.DM.ones(self.n_state, self.T_HORIZON + 1)
        self.ubx_X = ca.inf * ca.DM.ones(self.n_state, self.T_HORIZON + 1)
        
        # 控制变量界限  
        self.lbx_U = ca.repmat(ca.DM(self.u_min).reshape((-1, 1)), 1, self.T_HORIZON)
        self.ubx_U = ca.repmat(ca.DM(self.u_max).reshape((-1, 1)), 1, self.T_HORIZON)
        
        # 合并界限
        self.lbx = ca.vertcat(
            ca.reshape(self.lbx_X, -1, 1),
            ca.reshape(self.lbx_U, -1, 1)
        )
        self.ubx = ca.vertcat(
            ca.reshape(self.ubx_X, -1, 1),
            ca.reshape(self.ubx_U, -1, 1)
        )
        
        # 约束界限
        n_initial_constraints = self.n_state
        n_dynamics_constraints = self.T_HORIZON * self.n_state
        n_cbf_constraints = len(self.obstacles) * self.T_HORIZON  # CBF梯度约束
        # 修改：所有状态的安全约束 (T_HORIZON + 1) 个时间步
        n_safety_constraints = len(self.obstacles) * (self.T_HORIZON + 1)  
        
        total_constraints = (n_initial_constraints + n_dynamics_constraints + 
                        n_cbf_constraints + n_safety_constraints)
        
        print(f"Constraint breakdown:")
        print(f"  Initial: {n_initial_constraints}")
        print(f"  Dynamics: {n_dynamics_constraints}")
        print(f"  CBF (gradient): {n_cbf_constraints}")
        print(f"  Safety (all states): {n_safety_constraints}")
        print(f"  Total: {total_constraints}")
        
        # 设置约束界限
        self.lbg = []
        self.ubg = []
        
        # 1. 初始条件约束 (等式)
        self.lbg.extend([0.0] * n_initial_constraints)
        self.ubg.extend([0.0] * n_initial_constraints)
        
        # 2. 动力学约束 (等式)
        self.lbg.extend([0.0] * n_dynamics_constraints)
        self.ubg.extend([0.0] * n_dynamics_constraints)
        
        # 3. CBF梯度约束 (不等式 >= 0)
        self.lbg.extend([0.0] * n_cbf_constraints)
        self.ubg.extend([ca.inf] * n_cbf_constraints)

        self.lbg = ca.DM(self.lbg)
        self.ubg = ca.DM(self.ubg)
    
    def _quadrotor_dynamics(self, x, u):
        """四旋翼动力学模型"""
        dt = self.DT
        x_next = x + dt * self._quadrotor_dynamics_rhs(x, u)
        
        # 确保四元数归一化
        quat = x_next[kQuatW:kQuatZ+1]
        quat_norm = ca.sqrt(ca.sumsqr(quat))
        quat_norm = ca.fmax(quat_norm, 1e-8)
        x_next[kQuatW:kQuatZ+1] = quat / quat_norm
        
        return x_next
    
    def _quadrotor_dynamics_rhs(self, state, action):
        """四旋翼动力学右端项"""
        thrust = action[0]
        wx = action[1] 
        wy = action[2]
        wz = action[3]
        
        dstate = ca.SX.zeros(self.n_state)
        
        # 位置导数 = 速度
        dstate[kPosX] = state[kVelX]
        dstate[kPosY] = state[kVelY] 
        dstate[kPosZ] = state[kVelZ]
        
        # 四元数导数
        qw = state[kQuatW]
        qx = state[kQuatX]
        qy = state[kQuatY] 
        qz = state[kQuatZ]
        
        dstate[kQuatW] = 0.5 * (-wx*qx - wy*qy - wz*qz)
        dstate[kQuatX] = 0.5 * ( wx*qw + wz*qy - wy*qz)
        dstate[kQuatY] = 0.5 * ( wy*qw - wz*qx + wx*qz)
        dstate[kQuatZ] = 0.5 * ( wz*qw + wy*qx - wx*qy)
        
        # 速度导数
        dstate[kVelX] = 2 * (qw*qy + qx*qz) * thrust
        dstate[kVelY] = 2 * (qy*qz - qw*qx) * thrust  
        dstate[kVelZ] = (qw*qw - qx*qx - qy*qy + qz*qz) * thrust - self.quad._gz
        
        return dstate
    
    def run(self, machine_state, human_actions, human_states):
        """运行MPC求解器
        Args:
            machine_state: 当前机器状态 [n_state]
            human_actions: 人类控制输入序列 [T_HORIZON, n_ctrl] 
            human_states: 人类目标状态序列 [T_HORIZON, n_state]
        Returns:
            optimal_control: 最优控制输入 [n_ctrl]
        """
        # 确保输入维度正确
        x0 = np.array(machine_state, dtype=np.float64)
        print(self._barrier_function(x0, self.obstacles[0]))
        
        # 处理human_actions - 确保有T_HORIZON个控制输入
        if len(human_actions.shape) == 1:
            # 如果只有一个控制输入，复制T_HORIZON次
            u_human = np.tile(human_actions, (self.T_HORIZON, 1))
        else:
            u_human = np.array(human_actions, dtype=np.float64)
            
        # 确保human_actions有正确的维度
        if u_human.shape[0] < self.T_HORIZON:
            # 如果不够，用最后一个重复填充
            last_action = u_human[-1] if len(u_human) > 0 else np.zeros(self.n_ctrl)
            padding = np.tile(last_action, (self.T_HORIZON - u_human.shape[0], 1))
            u_human = np.vstack([u_human, padding])
        elif u_human.shape[0] > self.T_HORIZON:
            # 如果太多，截取前T_HORIZON个
            u_human = u_human[:self.T_HORIZON]
            
        # 处理human_states - 确保有T_HORIZON个状态
        x_human = np.array(human_states, dtype=np.float64)
        if x_human.shape[0] < self.T_HORIZON:
            # 如果不够，用最后一个重复填充
            last_state = x_human[-1] if len(x_human) > 0 else x0
            padding = np.tile(last_state, (self.T_HORIZON - x_human.shape[0], 1))
            x_human = np.vstack([x_human, padding])
        elif x_human.shape[0] > self.T_HORIZON:
            # 如果太多，截取前T_HORIZON个  
            x_human = x_human[:self.T_HORIZON]
        
        # 构建参数向量: [初始状态, 人类控制序列, 人类状态序列]
        p = ca.vertcat(
            ca.DM(x0),
            ca.DM(u_human.flatten()),
            ca.DM(x_human.flatten())
        )
        
        # 初始化优化变量
        x0_opt = ca.repmat(ca.DM(x0).reshape((-1, 1)), 1, self.T_HORIZON + 1)
        u0_opt = ca.DM.zeros(self.n_ctrl, self.T_HORIZON)
        
        # 更好的初始猜测 - 用人类控制输入作为初始猜测
        for k in range(self.T_HORIZON):
            if k < u_human.shape[0]:
                u0_opt[:, k] = u_human[k]
            else:
                u0_opt[:, k] = [9.81, 0.0, 0.0, 0.0]  # 悬停
        
        x0_opt = ca.vertcat(
            ca.reshape(x0_opt, -1, 1),
            ca.reshape(u0_opt, -1, 1)
        )

        try:
            # 求解
            sol = self.solver(
                x0=x0_opt,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            # # 检查求解状态
            # if not self.solver.stats()['success']:
            #     print("Warning: MPC solver did not converge")
            #     # 返回人类第一个控制输入作为备用
            #     return u_human[0] if len(u_human) > 0 else np.array([9.81, 0.0, 0.0, 0.0])
            
            # 提取解
            u_opt = ca.reshape(sol['x'][self.n_state * (self.T_HORIZON + 1):], 
                              self.n_ctrl, self.T_HORIZON)
            
            # 返回第一个控制输入
            u_optimal = np.array(u_opt[:, 0]).flatten()
            
            return u_optimal
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            # 返回人类控制输入作为备用
            return u_human[0] if len(u_human) > 0 else np.array([9.81, 0.0, 0.0, 0.0])

# 测试函数
from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from shared_autonomy_history import RLHuman, HumanMPC
def test_assistive_mpc_integration():
    """测试AssistiveMPC与shared_autonomy.py的整合"""
    import collections
    
    # === 模拟环境和模型 ===
    def scale_to_env(a_norm, action_low, action_high):
        return (a_norm + 1.0) / 2.0 * (action_high - action_low) + action_low
    
    # === 初始化系统 ===
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]
    
    rlhuman = RLHuman(state_dim, action_dim)

    humanmodel = HumanMPC(goal_weights= args.goal_weights,
                          ctrl_weights= args.ctrl_weights, T_HORIZON=15)

    # 创建AssistiveMPC - 关键修改：T_HORIZON要与humanmodel一致
    assistivempc = AssistiveMPC(
        goal_weights=[1,1,1,1,1,1,1,1,1,1],
        ctrl_weights=[1,1,1,1],
        T_HORIZON=15,  # 使用较小的horizon用于MPC
        cbf_gamma=0.1
    )
    
    # === 仿真循环 ===
    step_idx = 0
    done = False
    obs_dict, _ = env.reset()
    # 添加障碍物
    machine_state = obs_dict["machine"]
    assistivempc.add_obstacle(x_obs=machine_state[-3], y_obs=machine_state[-2], z_obs=machine_state[-1], radius=1)
    
    state = obs_dict["human"]
    
    states = collections.deque([state[:10]] * 3, maxlen=3)
    actions = collections.deque([np.zeros(action_dim)] * 3, maxlen=3)
    
    state_history = []
    control_history = []
    
    print("开始仿真...")
    while not done and step_idx < args.max_steps:  
        print(f"Step {step_idx}")
        
        # 1. RL生成动作
        a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
        env_act = scale_to_env(a_norm, action_low, action_high)
        
        # 2. 更新状态和动作历史
        states.append(state[:10])
        actions.append(env_act)
        
        # 3. 人类模型预测
        aim_goal = humanmodel.run(np.array(states), np.array(actions))
        x, u = humanmodel.step(state[:10], aim_goal)  # x: [50, 10], u: [50, 4]
        x=np.squeeze(x, axis=1)
        u=np.squeeze(u, axis=1)
        
        # 4. AssistiveMPC - 关键修改：使用MPC的T_HORIZON
        mpc_horizon = assistivempc.T_HORIZON
        
        # 截取或扩展到MPC的horizon
        human_states_mpc = x[:mpc_horizon]  # [T_HORIZON, 10]
        human_actions_mpc = np.tile(env_act, (mpc_horizon, 1))  # [T_HORIZON, 4]

        # 5. 运行AssistiveMPC
        try:
            assistive_action = assistivempc.run(
                machine_state=state[:10],           # 当前机器状态
                human_actions=human_actions_mpc,    # 人类控制序列 [T_HORIZON, 4]
                human_states=human_states_mpc,       # 人类状态序列 [T_HORIZON, 10]
            )
            
            print(f"  Original action: {env_act}")
            print(f"  Assistive action: {assistive_action}")
            # 使用assistive action
            env_act = assistive_action
            
        except Exception as e:
            print(f"  AssistiveMPC failed: {e}")
            # 使用原始动作
            pass
        
        # 6. 环境步进
        obs_dict, _, done, _ = env.step(env_act)
        next_state = obs_dict["human"]
        
        # 记录历史
        state_history.append(state[:10].copy())
        control_history.append(env_act.copy())
        
        state = next_state
        step_idx += 1
    
    print(f"仿真完成，总步数: {step_idx}")
    
    # 简单可视化
    if len(state_history) > 0:
        state_history = np.array(state_history)
        control_history = np.array(control_history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 位置轨迹
        ax1.plot(state_history[:, 0], label='X')
        ax1.plot(state_history[:, 1], label='Y') 
        ax1.plot(state_history[:, 2], label='Z')
        ax1.set_title('Position Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # 速度
        ax2.plot(state_history[:, 7], label='Vx')
        ax2.plot(state_history[:, 8], label='Vy')
        ax2.plot(state_history[:, 9], label='Vz')
        ax2.set_title('Velocity')
        ax2.legend()
        ax2.grid(True)
        
        # 控制输入
        ax3.plot(control_history[:, 0], label='Thrust')
        ax3.plot(control_history[:, 1], label='wx')
        ax3.plot(control_history[:, 2], label='wy')
        ax3.plot(control_history[:, 3], label='wz')
        ax3.set_title('Control Inputs')
        ax3.legend()
        ax3.grid(True)
        
        # 3D轨迹
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2])
        ax4.scatter(0, 0, 0, c='green', s=100, label='Start')
        ax4.scatter(state_history[-1, 0], state_history[-1, 1], state_history[-1, 2], 
                   c='red', s=100, label='End')
        
        # 绘制障碍物
        obs = assistivempc.obstacles[0]
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x_sphere = obs['radius'] * np.outer(np.cos(u), np.sin(v)) + obs['x']
        y_sphere = obs['radius'] * np.outer(np.sin(u), np.sin(v)) + obs['y']
        z_sphere = obs['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obs['z']
        ax4.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
        
        # 设置坐标轴等比例
        # 获取数据范围
        x_data = state_history[:, 0]
        y_data = state_history[:, 1]
        z_data = state_history[:, 2]
        
        # 包括障碍物在内的数据范围
        x_min = min(x_data.min(), obs['x'] - obs['radius'])
        x_max = max(x_data.max(), obs['x'] + obs['radius'])
        y_min = min(y_data.min(), obs['y'] - obs['radius'])
        y_max = max(y_data.max(), obs['y'] + obs['radius'])
        z_min = min(z_data.min(), obs['z'] - obs['radius'])
        z_max = max(z_data.max(), obs['z'] + obs['radius'])
        
        # 计算最大范围
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
        
        # 计算中心点
        x_center = (x_max + x_min) / 2.0
        y_center = (y_max + y_min) / 2.0
        z_center = (z_max + z_min) / 2.0
        
        # 设置等比例的坐标轴范围
        ax4.set_xlim(x_center - max_range, x_center + max_range)
        ax4.set_ylim(y_center - max_range, y_center + max_range)
        ax4.set_zlim(z_center - max_range, z_center + max_range)
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('3D Trajectory with Obstacles')
        ax4.legend()
        
        # 绘制障碍物
        obs = assistivempc.obstacles[0]
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x_sphere = obs['radius'] * np.outer(np.cos(u), np.sin(v)) + obs['x']
        y_sphere = obs['radius'] * np.outer(np.sin(u), np.sin(v)) + obs['y']
        z_sphere = obs['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obs['z']
        ax4.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('3D Trajectory with Obstacles')
        
        plt.tight_layout()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections

def evaluate_assistive_performance(num_episodes=100, intervention_threshold=0.2, max_steps=1000):
    """
    评估有/无机器辅助的性能对比
    
    Args:
        num_episodes: 仿真轮数
        intervention_threshold: 介入阈值（动作改变超过此比例算作介入）
        max_steps: 每轮最大步数
    
    Returns:
        dict: 包含所有统计指标的字典
    """
    
    def scale_to_env(a_norm, action_low, action_high):
        return (a_norm + 1.0) / 2.0 * (action_high - action_low) + action_low
    
    def compute_energy_consumption(action):
        """计算能量消耗 - 使用动作的二范数平方"""
        return np.sum(action**2)
    
    def compute_action_magnitude(action):
        """计算动作大小 - 使用二范数"""
        return np.linalg.norm(action)
    
    def compute_distance_to_obstacle(position, obstacle_pos):
        """计算到障碍物的距离"""
        return np.linalg.norm(position - obstacle_pos)
    
    def is_intervention(human_action, machine_action, threshold):
        """判断是否发生介入"""
        relative_change = np.linalg.norm(machine_action - human_action) / (np.linalg.norm(human_action) + 1e-8)
        return relative_change > threshold
    
    # 初始化统计数据
    results = {
        # 无辅助情况
        'no_assist': {
            'success_rate': 0,
            'min_obstacle_distances': [],
            'avg_action_magnitude': [],
            'avg_energy_consumption': [],
            'episode_lengths': [],
            'final_gate_indices': []
        },
        # 有辅助情况  
        'with_assist': {
            'success_rate': 0,
            'min_obstacle_distances': [],
            'avg_action_magnitude': [],
            'avg_energy_consumption': [],
            'intervention_frequencies': [],
            'avg_intervention_magnitude': [],
            'episode_lengths': [],
            'final_gate_indices': []
        }
    }
    
    print("开始评估性能...")
    print(f"总仿真轮数: {num_episodes}")
    print(f"介入阈值: {intervention_threshold}")
    
    # === 初始化系统组件 ===
    from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
    from shared_autonomy_history import RLHuman, HumanMPC
    from Personalized_SA.config.config import args

    env = QuadrotorRaceEnv(dt=0.01)  # 不使用可视化
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]
    
    rlhuman = RLHuman(state_dim, action_dim)
    humanmodel = HumanMPC(goal_weights=args.goal_weights,
                         ctrl_weights=args.ctrl_weights, T_HORIZON=15)
    
    # 创建AssistiveMPC
    assistivempc = AssistiveMPC(
        goal_weights=[1,1,1,1,1,1,1,1,1,1],
        ctrl_weights=[1,1,1,1],
        T_HORIZON=15,
        cbf_gamma=0.1
    )
    
    # === 运行无辅助仿真 ===
    print("\n=== 运行无辅助仿真 ===")
    no_assist_successes = 0
    global_min_obstacle_dist_no_assist = float('inf')  # 全局最小障碍物距离
    
    for episode in tqdm(range(num_episodes), desc="无辅助"):
        # 重置环境
        obs_dict, _ = env.reset(seed=episode)
        machine_state = obs_dict["machine"]
        state = obs_dict["human"]
        
        # 记录该轮的统计数据
        min_obstacle_dist = float('inf')
        action_magnitudes = []
        energy_consumptions = []
        
        # 无辅助情况不需要维护历史记录
        
        done = False
        step_count = 0
        crash = False

        while not done and step_count < max_steps:
            # 1. RL生成动作
            a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
            env_act = scale_to_env(a_norm, action_low, action_high)
            
            # 2. 直接执行原始人类动作（无辅助，无需预测）
            obs_dict, _, done, info = env.step(env_act)
            next_state = obs_dict["human"]
            
            # 3. 统计数据
            current_pos = state[:3]
            obstacle_pos = machine_state[-3:]  # 障碍物位置
            dist_to_obstacle = compute_distance_to_obstacle(current_pos, obstacle_pos)

            if dist_to_obstacle < 1:
                crash = True

            min_obstacle_dist = min(min_obstacle_dist, dist_to_obstacle)
            
            action_magnitudes.append(compute_action_magnitude(env_act))
            energy_consumptions.append(compute_energy_consumption(env_act))
            
            state = next_state
            step_count += 1
        
        # 4. 记录该轮结果
        if done and info.get('termination') == 'all_gates_passed' and not crash:
            no_assist_successes += 1
            
        results['no_assist']['min_obstacle_distances'].append(min_obstacle_dist)
        global_min_obstacle_dist_no_assist = min(global_min_obstacle_dist_no_assist, min_obstacle_dist)  # 更新全局最小值
        results['no_assist']['avg_action_magnitude'].append(np.mean(action_magnitudes))
        results['no_assist']['avg_energy_consumption'].append(np.mean(energy_consumptions))
        results['no_assist']['episode_lengths'].append(step_count)
        results['no_assist']['final_gate_indices'].append(env.current_gate_idx)
    
    results['no_assist']['success_rate'] = no_assist_successes / num_episodes
    
    # === 运行有辅助仿真 ===
    print("\n=== 运行有辅助仿真 ===")
    with_assist_successes = 0
    global_min_obstacle_dist_with_assist = float('inf')  # 全局最小障碍物距离
    
    for episode in tqdm(range(num_episodes), desc="有辅助"):
        # 重置环境
        obs_dict, _ = env.reset(seed=episode)  # 使用相同种子确保公平对比
        machine_state = obs_dict["machine"]
        state = obs_dict["human"]
        
        # 为AssistiveMPC添加障碍物
        assistivempc.obstacles = []  # 清空之前的障碍物
        obstacle_pos = machine_state[-3:]
        assistivempc.add_obstacle(x_obs=obstacle_pos[0], y_obs=obstacle_pos[1], 
                                z_obs=obstacle_pos[2], radius=1.0)
        
        # 记录该轮的统计数据
        min_obstacle_dist = float('inf')
        action_magnitudes = []
        energy_consumptions = []
        interventions = []
        intervention_magnitudes = []
        
        # 初始化历史
        states = collections.deque([state[:10]] * 3, maxlen=3)
        actions = collections.deque([np.zeros(action_dim)] * 3, maxlen=3)
        
        done = False
        step_count = 0
        crash = False
        
        while not done and step_count < max_steps:
            # 1. RL生成动作
            a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
            human_action = scale_to_env(a_norm, action_low, action_high)
            
            # 2. 更新历史
            states.append(state[:10])
            actions.append(human_action)
            
            # 3. 人类模型预测
            aim_goal = humanmodel.run(np.array(states), np.array(actions))
            x, u = humanmodel.step(state[:10], aim_goal)
            x = np.squeeze(x, axis=1)
            u = np.squeeze(u, axis=1)
            
            # 4. AssistiveMPC生成辅助动作
            try:
                mpc_horizon = assistivempc.T_HORIZON
                human_states_mpc = x[:mpc_horizon]
                human_actions_mpc = np.tile(human_action, (mpc_horizon, 1))
                
                assistive_action = assistivempc.run(
                    machine_state=state[:10],
                    human_actions=human_actions_mpc,
                    human_states=human_states_mpc,
                )
                
                # 5. 判断是否介入
                is_intervened = is_intervention(human_action, assistive_action, intervention_threshold)
                interventions.append(is_intervened)
                
                if is_intervened:
                    intervention_magnitude = np.linalg.norm(assistive_action - human_action)
                    intervention_magnitudes.append(intervention_magnitude)
                
                # 使用辅助动作
                final_action = assistive_action
                
            except Exception as e:
                # MPC失败时使用原始人类动作
                interventions.append(False)
                final_action = human_action
            
            # 6. 执行动作
            obs_dict, _, done, info = env.step(final_action)
            next_state = obs_dict["human"]
            
            # 7. 统计数据
            current_pos = state[:3]
            dist_to_obstacle = compute_distance_to_obstacle(current_pos, obstacle_pos)

            if dist_to_obstacle < 1:
                crash = True

            min_obstacle_dist = min(min_obstacle_dist, dist_to_obstacle)
            
            action_magnitudes.append(compute_action_magnitude(final_action))
            energy_consumptions.append(compute_energy_consumption(final_action))
            
            state = next_state
            step_count += 1
        
        # 8. 记录该轮结果
        if done and info.get('termination') == 'all_gates_passed' and not crash:
            with_assist_successes += 1
            
        results['with_assist']['min_obstacle_distances'].append(min_obstacle_dist)
        global_min_obstacle_dist_with_assist = min(global_min_obstacle_dist_with_assist, min_obstacle_dist)  # 更新全局最小值
        results['with_assist']['avg_action_magnitude'].append(np.mean(action_magnitudes))
        results['with_assist']['avg_energy_consumption'].append(np.mean(energy_consumptions))
        results['with_assist']['intervention_frequencies'].append(np.mean(interventions))
        results['with_assist']['avg_intervention_magnitude'].append(
            np.mean(intervention_magnitudes) if intervention_magnitudes else 0.0
        )
        results['with_assist']['episode_lengths'].append(step_count)
        results['with_assist']['final_gate_indices'].append(env.current_gate_idx)
    
    results['with_assist']['success_rate'] = with_assist_successes / num_episodes
    
    # === 计算汇总统计 ===
    print("\n" + "="*50)
    print("=== 仿真结果汇总 ===")
    print("="*50)
    
    # 无辅助结果
    print(f"\n【无辅助结果】")
    print(f"成功轮数: {no_assist_successes}/{num_episodes}")
    print(f"成功率: {results['no_assist']['success_rate']:.3f}")
    print(f"平均最近障碍物距离: {np.mean(results['no_assist']['min_obstacle_distances']):.3f}m")
    print(f"全局最小障碍物距离: {global_min_obstacle_dist_no_assist:.3f}m")
    print(f"平均动作大小: {np.mean(results['no_assist']['avg_action_magnitude']):.3f}")
    print(f"平均能量消耗: {np.mean(results['no_assist']['avg_energy_consumption']):.3f}")
    print(f"平均仿真步数: {np.mean(results['no_assist']['episode_lengths']):.1f}")
    print(f"平均最终门数: {np.mean(results['no_assist']['final_gate_indices']):.2f}/{env.num_gates}")
    
    # 有辅助结果
    print(f"\n【有辅助结果】")
    print(f"成功轮数: {with_assist_successes}/{num_episodes}")
    print(f"成功率: {results['with_assist']['success_rate']:.3f}")
    print(f"平均最近障碍物距离: {np.mean(results['with_assist']['min_obstacle_distances']):.3f}m")
    print(f"全局最小障碍物距离: {global_min_obstacle_dist_with_assist:.3f}m")
    print(f"平均动作大小: {np.mean(results['with_assist']['avg_action_magnitude']):.3f}")
    print(f"平均能量消耗: {np.mean(results['with_assist']['avg_energy_consumption']):.3f}")
    print(f"平均介入频率: {np.mean(results['with_assist']['intervention_frequencies']):.3f}")
    print(f"平均介入动作大小: {np.mean(results['with_assist']['avg_intervention_magnitude']):.3f}")
    print(f"平均仿真步数: {np.mean(results['with_assist']['episode_lengths']):.1f}")
    print(f"平均最终门数: {np.mean(results['with_assist']['final_gate_indices']):.2f}/{env.num_gates}")
    
    print("\n" + "="*50)
    print("=== 性能对比总结 ===")
    print("="*50)
    print(f"{'指标':<20} {'无辅助':<15} {'有辅助':<15} {'改进':<15}")
    print("-" * 65)
    print(f"{'成功率':<20} {results['no_assist']['success_rate']:<15.3f} {results['with_assist']['success_rate']:<15.3f} {results['with_assist']['success_rate'] - results['no_assist']['success_rate']:+.3f}")
    print(f"{'平均最近障碍距离':<20} {np.mean(results['no_assist']['min_obstacle_distances']):<15.3f} {np.mean(results['with_assist']['min_obstacle_distances']):<15.3f} {np.mean(results['with_assist']['min_obstacle_distances']) - np.mean(results['no_assist']['min_obstacle_distances']):+.3f}")
    print(f"{'全局最小障碍距离':<20} {global_min_obstacle_dist_no_assist:<15.3f} {global_min_obstacle_dist_with_assist:<15.3f} {global_min_obstacle_dist_with_assist - global_min_obstacle_dist_no_assist:+.3f}")
    print(f"{'平均动作大小':<20} {np.mean(results['no_assist']['avg_action_magnitude']):<15.3f} {np.mean(results['with_assist']['avg_action_magnitude']):<15.3f} {np.mean(results['with_assist']['avg_action_magnitude']) - np.mean(results['no_assist']['avg_action_magnitude']):+.3f}")
    print(f"{'平均能量消耗':<20} {np.mean(results['no_assist']['avg_energy_consumption']):<15.3f} {np.mean(results['with_assist']['avg_energy_consumption']):<15.3f} {np.mean(results['with_assist']['avg_energy_consumption']) - np.mean(results['no_assist']['avg_energy_consumption']):+.3f}")
    print("-" * 65)
    print(f"平均介入频率: {np.mean(results['with_assist']['intervention_frequencies']):.3f}")
    print(f"平均介入动作大小: {np.mean(results['with_assist']['avg_intervention_magnitude']):.3f}")
    print(f"介入阈值设置: {intervention_threshold}")
    print("="*50)
    
    # 添加汇总统计到结果中
    results['summary'] = {
        'success_rate_improvement': results['with_assist']['success_rate'] - results['no_assist']['success_rate'],
        'avg_min_obstacle_dist_no_assist': np.mean(results['no_assist']['min_obstacle_distances']),
        'avg_min_obstacle_dist_with_assist': np.mean(results['with_assist']['min_obstacle_distances']),
        'global_min_obstacle_dist_no_assist': global_min_obstacle_dist_no_assist,
        'global_min_obstacle_dist_with_assist': global_min_obstacle_dist_with_assist,
        'avg_action_magnitude_no_assist': np.mean(results['no_assist']['avg_action_magnitude']),
        'avg_action_magnitude_with_assist': np.mean(results['with_assist']['avg_action_magnitude']),
        'avg_energy_consumption_no_assist': np.mean(results['no_assist']['avg_energy_consumption']),
        'avg_energy_consumption_with_assist': np.mean(results['with_assist']['avg_energy_consumption']),
        'avg_intervention_frequency': np.mean(results['with_assist']['intervention_frequencies']),
        'avg_intervention_magnitude': np.mean(results['with_assist']['avg_intervention_magnitude']),
        'intervention_threshold': intervention_threshold
    }
    
    return results

def plot_evaluation_results(results):
    """绘制评估结果的可视化图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 成功率对比
    categories = ['No Assist', 'With Assist']
    success_rates = [results['no_assist']['success_rate'], results['with_assist']['success_rate']]
    bars1 = ax1.bar(categories, success_rates, color=['lightblue', 'lightgreen'])
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate Comparison')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 最近障碍物距离分布
    ax2.hist(results['no_assist']['min_obstacle_distances'], alpha=0.7, label='No Assist', bins=20)
    ax2.hist(results['with_assist']['min_obstacle_distances'], alpha=0.7, label='With Assist', bins=20)
    
    # 添加全局最小值的垂直线
    global_min_no_assist = results['summary']['global_min_obstacle_dist_no_assist']
    global_min_with_assist = results['summary']['global_min_obstacle_dist_with_assist']
    ax2.axvline(global_min_no_assist, color='blue', linestyle='--', 
                label=f'No Assist Global Min: {global_min_no_assist:.3f}m')
    ax2.axvline(global_min_with_assist, color='orange', linestyle='--',
                label=f'With Assist Global Min: {global_min_with_assist:.3f}m')
    
    ax2.set_xlabel('Minimum Obstacle Distance (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Minimum Obstacle Distance Distribution')
    ax2.legend()
    
    # 3. 能量消耗对比
    energy_data = [results['no_assist']['avg_energy_consumption'], 
                   results['with_assist']['avg_energy_consumption']]
    bp = ax3.boxplot(energy_data, labels=['No Assist', 'With Assist'])
    ax3.set_ylabel('Average Energy Consumption')
    ax3.set_title('Energy Consumption Distribution')
    
    # 4. 介入统计
    intervention_freq = results['with_assist']['intervention_frequencies']
    intervention_mag = results['with_assist']['avg_intervention_magnitude']
    
    ax4_twin = ax4.twinx()
    bars4_1 = ax4.bar(range(len(intervention_freq)), intervention_freq, alpha=0.7, 
                     color='orange', label='Intervention Frequency')
    bars4_2 = ax4_twin.bar(range(len(intervention_mag)), intervention_mag, alpha=0.7, 
                          color='red', label='Intervention Magnitude')
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Intervention Frequency', color='orange')
    ax4_twin.set_ylabel('Intervention Magnitude', color='red')
    ax4.set_title('Intervention Frequency and Magnitude')
    
    plt.tight_layout()
    plt.show()


def evaluate_different_cbf_gammas(cbf_gamma_list=[0.05, 0.1, 0.2, 0.5], 
                                  num_episodes=10, 
                                  max_steps=1000,
                                  intervention_threshold=0.2):
    """
    评估不同cbf_gamma值的效果
    
    Args:
        cbf_gamma_list: 要测试的cbf_gamma值列表
        num_episodes: 每个gamma值的测试轮数
        max_steps: 每轮最大步数
        intervention_threshold: 介入阈值
    
    Returns:
        dict: 包含所有gamma值结果的字典
    """
    
    # 复用现有函数
    def scale_to_env(a_norm, action_low, action_high):
        return (a_norm + 1.0) / 2.0 * (action_high - action_low) + action_low
    
    def is_intervention(human_action, machine_action, threshold):
        """判断是否发生介入"""
        relative_change = np.linalg.norm(machine_action - human_action) / (np.linalg.norm(human_action) + 1e-8)
        return relative_change > threshold
    
    # 初始化环境和模型
    from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
    from shared_autonomy_history import RLHuman, HumanMPC
    
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]
    
    rlhuman = RLHuman(state_dim, action_dim)
    humanmodel = HumanMPC(goal_weights=args.goal_weights,
                         ctrl_weights=args.ctrl_weights, T_HORIZON=15)
    
    # 存储所有结果
    all_results = {}
    
    # 对每个cbf_gamma值进行测试
    for gamma in cbf_gamma_list:
        print(f"\n=== 测试 cbf_gamma = {gamma} ===")
        
        # 创建新的AssistiveMPC实例
        assistivempc = AssistiveMPC(
            goal_weights=[1,1,1,1,1,1,1,1,1,1],
            ctrl_weights=[1,1,1,1],
            T_HORIZON=15,
            cbf_gamma=gamma
        )
        
        # 存储该gamma的结果
        gamma_results = {
            'trajectories': [],  # 存储每个episode的轨迹
            'interventions': [],  # 存储介入信息
            'min_obstacle_distances': [],
            'success_count': 0
        }
        
        # 运行多个episode
        for episode in tqdm(range(num_episodes), desc=f"gamma={gamma}"):
            # 重置环境
            obs_dict, _ = env.reset(seed=episode)
            machine_state = obs_dict["machine"]
            state = obs_dict["human"]
            
            # 添加障碍物
            assistivempc.obstacles = []
            obstacle_pos = machine_state[-3:]
            assistivempc.add_obstacle(x_obs=obstacle_pos[0], y_obs=obstacle_pos[1], 
                                    z_obs=obstacle_pos[2], radius=1.0)
            
            # 存储轨迹和介入数据
            trajectory = []
            intervention_data = []
            min_obstacle_dist = float('inf')
            
            # 初始化历史
            states = collections.deque([state[:10]] * 3, maxlen=3)
            actions = collections.deque([np.zeros(action_dim)] * 3, maxlen=3)
            
            done = False
            step_count = 0
            crash = False
            
            while not done and step_count < max_steps:
                # 1. RL生成动作
                a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
                human_action = scale_to_env(a_norm, action_low, action_high)
                
                # 2. 更新历史
                states.append(state[:10])
                actions.append(human_action)
                
                # 3. 人类模型预测
                aim_goal = humanmodel.run(np.array(states), np.array(actions))
                x, u = humanmodel.step(state[:10], aim_goal)
                x = np.squeeze(x, axis=1)
                u = np.squeeze(u, axis=1)
                
                # 4. AssistiveMPC
                try:
                    mpc_horizon = assistivempc.T_HORIZON
                    human_states_mpc = x[:mpc_horizon]
                    human_actions_mpc = np.tile(human_action, (mpc_horizon, 1))
                    
                    assistive_action = assistivempc.run(
                        machine_state=state[:10],
                        human_actions=human_actions_mpc,
                        human_states=human_states_mpc,
                    )
                    
                    # 计算介入大小
                    is_intervened = is_intervention(human_action, assistive_action, intervention_threshold)
                    intervention_magnitude = np.linalg.norm(assistive_action - human_action)
                    
                    final_action = assistive_action
                    
                except Exception as e:
                    intervention_magnitude = 0.0
                    final_action = human_action
                
                # 5. 执行动作
                obs_dict, _, done, info = env.step(final_action)
                next_state = obs_dict["human"]
                
                # 6. 记录数据
                current_pos = state[:3].copy()
                trajectory.append({
                    'x': current_pos[0],
                    'y': current_pos[1],
                    'z': current_pos[2],
                    'intervention_magnitude': intervention_magnitude
                })
                
                # 计算到障碍物的距离
                dist_to_obstacle = np.linalg.norm(current_pos - obstacle_pos)
                if dist_to_obstacle < 1:
                    crash = True
                min_obstacle_dist = min(min_obstacle_dist, dist_to_obstacle)
                
                state = next_state
                step_count += 1
            
            # 记录episode结果
            gamma_results['trajectories'].append(trajectory)
            gamma_results['min_obstacle_distances'].append(min_obstacle_dist)
            if done and info.get('termination') == 'all_gates_passed' and not crash:
                gamma_results['success_count'] += 1
        
        gamma_results['success_rate'] = gamma_results['success_count'] / num_episodes
        all_results[gamma] = gamma_results
        
        print(f"cbf_gamma={gamma}: 成功率={gamma_results['success_rate']:.3f}, "
              f"平均最小障碍距离={np.mean(gamma_results['min_obstacle_distances']):.3f}")
    
    return all_results

def plot_cbf_gamma_comparison(results, cbf_gamma_list):
    """
    绘制不同cbf_gamma的对比结果
    创建一个4行的竖向图：x位置、y位置、z位置、介入大小
    """
    if 'dt' in results:
        dt = results['dt']
        xlabel = 'Time (s)'
    else:
        dt = 1 # 如果没有提供dt，则默认时间步长为1
        xlabel = 'Time Steps'

    # 使用高对比度的颜色
        colors = [
        '#FF0000',  # 红色
        "#058805",  # 绿色
        '#FFA600',  # 黄色
    ]
    if len(cbf_gamma_list) > len(colors):
        colors = plt.cm.viridis(np.linspace(0, 1, len(cbf_gamma_list)))
    
    # 您选择的 figsize
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

    lines, labels = [], []

    # 遍历数据并绘图
    for idx, gamma in enumerate(cbf_gamma_list):
        gamma_data = results.get(gamma)
        if not gamma_data or not gamma_data.get('trajectories'):
            continue
            
        trajectory = gamma_data['trajectories'][0]
        num_steps = len(trajectory)
        time_axis = np.arange(num_steps) * dt
        
        x_positions = [t['x'] for t in trajectory]
        y_positions = [t['y'] for t in trajectory]
        z_positions = [t['z'] for t in trajectory]
        interventions = [t['intervention_magnitude'] for t in trajectory] 
            
        label_text = f'γ={gamma}'

        line, = ax1.plot(time_axis, x_positions, color=colors[idx], 
                         label=label_text, linewidth=2)
        ax2.plot(time_axis, y_positions, color=colors[idx], linewidth=2)
        ax3.plot(time_axis, z_positions, color=colors[idx], linewidth=2)
        
        ax4.plot(time_axis, interventions, color=colors[idx], 
                 marker='*', linestyle='None', markersize=5)
            
        lines.append(line)
        labels.append(label_text)

    # fig.suptitle('Quadrotor Trajectory Comparison for Different γ', fontsize=14, y=0.98)

    ax1.set_ylabel('x (m)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    ax3.set_ylabel('z (m)', fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    ax4.set_ylabel('Intervention', fontsize=11, fontweight='bold')
    ax4.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5) 

    # 调整图例
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=len(cbf_gamma_list), frameon=False, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    plt.show()

def plot_cbf_gamma_statistics(results, cbf_gamma_list):
    """
    绘制不同cbf_gamma的统计对比图
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 成功率对比
    success_rates = [results[gamma]['success_rate'] for gamma in cbf_gamma_list]
    bars1 = ax1.bar(range(len(cbf_gamma_list)), success_rates, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(cbf_gamma_list))))
    ax1.set_xlabel('CBF γ')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs CBF γ')
    ax1.set_xticks(range(len(cbf_gamma_list)))
    ax1.set_xticklabels([f'{gamma}' for gamma in cbf_gamma_list])
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # 2. 平均最小障碍物距离
    avg_min_distances = [np.mean(results[gamma]['min_obstacle_distances']) 
                        for gamma in cbf_gamma_list]
    ax2.plot(cbf_gamma_list, avg_min_distances, 'o-', markersize=10, linewidth=2)
    ax2.set_xlabel('CBF γ')
    ax2.set_ylabel('Average Minimum Obstacle Distance (m)')
    ax2.set_title('Safety Performance vs CBF γ')
    ax2.grid(True, alpha=0.3)
    
    # 3. 最小障碍物距离分布（箱线图）
    distance_data = [results[gamma]['min_obstacle_distances'] for gamma in cbf_gamma_list]
    bp = ax3.boxplot(distance_data, labels=[f'γ={gamma}' for gamma in cbf_gamma_list])
    ax3.set_ylabel('Minimum Obstacle Distance (m)')
    ax3.set_title('Distribution of Minimum Obstacle Distances')
    ax3.grid(True, alpha=0.3)
    
    # 4. 介入频率热力图（如果有多个episode）
    if len(results[cbf_gamma_list[0]]['trajectories']) > 1:
        intervention_matrix = []
        for gamma in cbf_gamma_list:
            gamma_interventions = []
            for traj in results[gamma]['trajectories'][:5]:  # 只显示前5个episode
                avg_intervention = np.mean([t['intervention_magnitude'] for t in traj])
                gamma_interventions.append(avg_intervention)
            intervention_matrix.append(gamma_interventions)
        
        im = ax4.imshow(intervention_matrix, aspect='auto', cmap='YlOrRd')
        ax4.set_yticks(range(len(cbf_gamma_list)))
        ax4.set_yticklabels([f'γ={gamma}' for gamma in cbf_gamma_list])
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('CBF γ')
        ax4.set_title('Average Intervention Magnitude Heatmap')
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.show()

import pickle
from pathlib import Path
def save_cbf_results(results, cbf_gamma_list, file_path="cbf_results.pkl"):
    """
    将 evaluate_different_cbf_gammas 返回的 results 与 cbf_gamma_list 一并保存
    """
    file_path = Path(file_path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)   # 若目录不存在则创建
    with file_path.open("wb") as f:
        # 一起打包，后续读取更方便
        pickle.dump({"results": results,
                     "cbf_gamma_list": cbf_gamma_list}, f)
    print(f"[√] 已保存到: {file_path}")

def load_cbf_results(file_path="cbf_results.pkl"):
    """
    读取之前保存的结果，返回 (results, cbf_gamma_list)
    """
    file_path = Path(file_path).expanduser().resolve()
    with file_path.open("rb") as f:
        data = pickle.load(f)
    print(f"[√] 已读取: {file_path}")
    return data["results"], data["cbf_gamma_list"]

if __name__ == "__main__":
    # test_assistive_mpc_integration()


    # # 运行评估
    # results = evaluate_assistive_performance(
    #     num_episodes=20,            # 仿真轮数
    #     intervention_threshold=0.2, # 介入阈值 
    #     max_steps=5000             # 每轮最大步数
    # )
    
    # # 绘制结果
    # plot_evaluation_results(results)


    # # 测试不同的cbf_gamma值
    # cbf_gamma_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    # print("开始测试不同的CBF-γ值...")
    # results = evaluate_different_cbf_gammas(
    #     cbf_gamma_list=cbf_gamma_list,
    #     num_episodes=5,  # 每个gamma测试5轮
    #     max_steps=5000,
    #     intervention_threshold=0.2
    # )
    
    # save_cbf_results(results, cbf_gamma_list, "cbf_results.pkl")

    results, cbf_gamma_list = load_cbf_results("cbf_results.pkl")

    # 绘制轨迹对比图
    print("\n绘制轨迹对比图...")
    plot_cbf_gamma_comparison(results, [0.1, 0.3, 0.5])
    
    # 绘制统计对比图
    print("\n绘制统计对比图...")
    plot_cbf_gamma_statistics(results, cbf_gamma_list)

