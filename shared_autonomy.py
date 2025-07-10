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
        obs_dict, reward, done, info = env.step(env_act)
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

if __name__ == "__main__":
    test_assistive_mpc_integration()