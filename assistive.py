import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import collections
from tqdm import tqdm

# 假设这些模块存在，如果不存在请相应调整import路径
from Personalized_SA.config.config import args
from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from shared_autonomy_history import RLHuman, HumanMPC

# 状态索引常量
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
        
        print(f"AssistiveMPC initialized with T_HORIZON={self.T_HORIZON}, gamma={self.cbf_gamma}")
        
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
        """设置CasADi优化器"""
        # 创建优化变量
        X = ca.SX.sym('X', self.n_state, self.T_HORIZON + 1)
        U = ca.SX.sym('U', self.n_ctrl, self.T_HORIZON)
        
        # 参数向量
        param_size = self.n_state + self.T_HORIZON * self.n_ctrl + self.T_HORIZON * self.n_state
        P = ca.SX.sym('P', param_size)
        
        # 创建目标函数
        obj = 0
        g = []
        
        # 初始条件约束
        g.append(X[:, 0] - P[:self.n_state])
        
        # 解析参数
        idx = self.n_state
        U_human = ca.reshape(P[idx:idx + self.T_HORIZON * self.n_ctrl], 
                            self.n_ctrl, self.T_HORIZON)
        idx += self.T_HORIZON * self.n_ctrl
        X_human = ca.reshape(P[idx:idx + self.T_HORIZON * self.n_state],
                            self.n_state, self.T_HORIZON)
        
        # 构建目标函数
        u_error = U[:, 0] - U_human[:, 0]
        Q_u_track = ca.diag(self.ctrl_weights)
        obj += ca.mtimes([u_error.T, Q_u_track, u_error])

        for k in range(self.T_HORIZON):
            # 状态跟踪项
            x_error = X[:, k] - X_human[:, k]  
            Q_x = ca.diag(self.goal_weights)
            obj += ca.mtimes([x_error.T, Q_x, x_error])
            
            # 控制努力项
            Q_u_reg = ca.diag(self.ctrl_weights * 0)  # rm3 = 0
            obj += ca.mtimes([U[:, k].T, Q_u_reg, U[:, k]])
            
            # 动力学约束
            x_next = self._quadrotor_dynamics(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
        
        # CBF约束
        for k in range(self.T_HORIZON):
            for obs in self.obstacles:
                h_curr = self._barrier_function(X[:, k], obs)
                h_next = self._barrier_function(X[:, k+1], obs)
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
        
        # 求解器选项
        opts = {
            'ipopt': {
                'max_iter': 100,
                'print_level': 0,
                'acceptable_tol': 1e-4,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
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
        n_cbf_constraints = len(self.obstacles) * self.T_HORIZON
        
        total_constraints = n_initial_constraints + n_dynamics_constraints + n_cbf_constraints
        
        # 设置约束界限
        self.lbg = []
        self.ubg = []
        
        # 初始条件约束 (等式)
        self.lbg.extend([0.0] * n_initial_constraints)
        self.ubg.extend([0.0] * n_initial_constraints)
        
        # 动力学约束 (等式)
        self.lbg.extend([0.0] * n_dynamics_constraints)
        self.ubg.extend([0.0] * n_dynamics_constraints)
        
        # CBF约束 (不等式 >= 0)
        self.lbg.extend([0.0] * n_cbf_constraints)
        self.ubg.extend([ca.inf] * n_cbf_constraints)

        self.lbg = ca.DM(self.lbg)
        self.ubg = ca.DM(self.ubg)
    
    def _quadrotor_dynamics(self, x, u):
        """四旋翼动力学模型"""
        dt = self.DT
        x_next = x + dt * self._quadrotor_dynamics_rhs(x, u)
        
        # 四元数归一化
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
        """运行MPC求解器"""
        x0 = np.array(machine_state, dtype=np.float64)
        
        # 处理human_actions
        if len(human_actions.shape) == 1:
            u_human = np.tile(human_actions, (self.T_HORIZON, 1))
        else:
            u_human = np.array(human_actions, dtype=np.float64)
            
        if u_human.shape[0] < self.T_HORIZON:
            last_action = u_human[-1] if len(u_human) > 0 else np.zeros(self.n_ctrl)
            padding = np.tile(last_action, (self.T_HORIZON - u_human.shape[0], 1))
            u_human = np.vstack([u_human, padding])
        elif u_human.shape[0] > self.T_HORIZON:
            u_human = u_human[:self.T_HORIZON]
            
        # 处理human_states
        x_human = np.array(human_states, dtype=np.float64)
        if x_human.shape[0] < self.T_HORIZON:
            last_state = x_human[-1] if len(x_human) > 0 else x0
            padding = np.tile(last_state, (self.T_HORIZON - x_human.shape[0], 1))
            x_human = np.vstack([x_human, padding])
        elif x_human.shape[0] > self.T_HORIZON:
            x_human = x_human[:self.T_HORIZON]
        
        # 构建参数向量
        p = ca.vertcat(
            ca.DM(x0),
            ca.DM(u_human.flatten()),
            ca.DM(x_human.flatten())
        )
        
        # 初始化优化变量
        x0_opt = ca.repmat(ca.DM(x0).reshape((-1, 1)), 1, self.T_HORIZON + 1)
        u0_opt = ca.DM.zeros(self.n_ctrl, self.T_HORIZON)
        
        for k in range(self.T_HORIZON):
            if k < u_human.shape[0]:
                u0_opt[:, k] = u_human[k]
            else:
                u0_opt[:, k] = [9.81, 0.0, 0.0, 0.0]
        
        x0_opt = ca.vertcat(
            ca.reshape(x0_opt, -1, 1),
            ca.reshape(u0_opt, -1, 1)
        )

        try:
            sol = self.solver(
                x0=x0_opt,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            u_opt = ca.reshape(sol['x'][self.n_state * (self.T_HORIZON + 1):], 
                              self.n_ctrl, self.T_HORIZON)
            
            u_optimal = np.array(u_opt[:, 0]).flatten()
            return u_optimal
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            return u_human[0] if len(u_human) > 0 else np.array([9.81, 0.0, 0.0, 0.0])


def run_comparison_simulation(max_steps=3000):
    """运行对比仿真：无辅助 vs 有辅助"""
    
    def scale_to_env(a_norm, action_low, action_high):
        return (a_norm + 1.0) / 2.0 * (action_high - action_low) + action_low
    
    # 初始化环境和模型
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]
    
    rlhuman = RLHuman(state_dim, action_dim)
    humanmodel = HumanMPC(goal_weights=args.goal_weights,
                         ctrl_weights=args.ctrl_weights, T_HORIZON=15)
    
    # 存储结果
    results = {
        'no_assist': {'trajectory': [], 'controls': [], 'obstacle_pos': None},
        'with_assist': {'trajectory': [], 'controls': [], 'obstacle_pos': None}
    }
    
    print("=== 运行无辅助仿真 ===")
    # 无辅助仿真
    obs_dict, _ = env.reset(seed=42)  # 固定种子确保一致性
    machine_state = obs_dict["machine"]
    state = obs_dict["human"]
    obstacle_pos = machine_state[-3:]
    results['no_assist']['obstacle_pos'] = obstacle_pos.copy()
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # RL生成动作
        a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
        env_act = scale_to_env(a_norm, action_low, action_high)
        
        # 记录轨迹
        results['no_assist']['trajectory'].append(state[:10].copy())
        results['no_assist']['controls'].append(env_act.copy())
        
        # 执行动作
        obs_dict, _, done, info = env.step(env_act)
        next_state = obs_dict["human"]
        
        state = next_state
        step_count += 1
    
    print(f"无辅助仿真完成，步数: {step_count}")
    
    print("=== 运行有辅助仿真 ===")
    # 有辅助仿真
    obs_dict, _ = env.reset(seed=42)  # 相同种子
    machine_state = obs_dict["machine"]
    state = obs_dict["human"]
    obstacle_pos = machine_state[-3:]
    results['with_assist']['obstacle_pos'] = obstacle_pos.copy()
    
    # 创建AssistiveMPC - gamma=0.1, rm3=0.1
    assistivempc = AssistiveMPC(
        goal_weights=[1,1,1,1,1,1,1,1,1,1],
        ctrl_weights=[1,1,1,1],
        T_HORIZON=15,
        cbf_gamma=0.1  # gamma = 0.1
    )
    assistivempc.add_obstacle(x_obs=obstacle_pos[0], y_obs=obstacle_pos[1], 
                             z_obs=obstacle_pos[2], radius=1.0)
    
    # 初始化历史
    states = collections.deque([state[:10]] * 3, maxlen=3)
    actions = collections.deque([np.zeros(action_dim)] * 3, maxlen=3)
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # RL生成动作
        a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
        human_action = scale_to_env(a_norm, action_low, action_high)
        
        # 更新历史
        states.append(state[:10])
        actions.append(human_action)
        
        # 人类模型预测
        aim_goal = humanmodel.run(np.array(states), np.array(actions))
        x, u = humanmodel.step(state[:10], aim_goal)
        x = np.squeeze(x, axis=1)
        u = np.squeeze(u, axis=1)
        
        # AssistiveMPC生成辅助动作
        try:
            mpc_horizon = assistivempc.T_HORIZON
            human_states_mpc = x[:mpc_horizon]
            human_actions_mpc = np.tile(human_action, (mpc_horizon, 1))
            
            assistive_action = assistivempc.run(
                machine_state=state[:10],
                human_actions=human_actions_mpc,
                human_states=human_states_mpc,
            )
            final_action = assistive_action
            
        except Exception as e:
            print(f"MPC failed at step {step_count}: {e}")
            final_action = human_action
        
        # 记录轨迹
        results['with_assist']['trajectory'].append(state[:10].copy())
        results['with_assist']['controls'].append(final_action.copy())
        
        # 执行动作
        obs_dict, _, done, info = env.step(final_action)
        next_state = obs_dict["human"]
        
        state = next_state
        step_count += 1
    
    print(f"有辅助仿真完成，步数: {step_count}")
    
    # 转换为numpy数组
    for key in results:
        results[key]['trajectory'] = np.array(results[key]['trajectory'])
        results[key]['controls'] = np.array(results[key]['controls'])
    
    return results


def save_simulation_results(results, file_path="simulation_results.pkl"):
    """保存仿真结果"""
    file_path = Path(file_path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with file_path.open("wb") as f:
        pickle.dump(results, f)
    
    print(f"[√] 仿真结果已保存到: {file_path}")


def load_simulation_results(file_path="simulation_results.pkl"):
    """读取仿真结果"""
    file_path = Path(file_path).expanduser().resolve()
    
    with file_path.open("rb") as f:
        results = pickle.load(f)
    
    print(f"[√] 仿真结果已读取: {file_path}")
    return results


def plot_separate_trajectories(results):
    """绘制两张分离的3D轨迹图，并标注碰撞点"""
    
    # 设置图形参数
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.0
    })
    
    # 定义碰撞检测函数
    def detect_collisions(trajectory, obstacle_pos, obstacle_radius=1.0):
        """
        检测轨迹中的碰撞点
        返回：(碰撞点索引, 距离数组, 是否发生碰撞)
        """
        # 计算每个轨迹点到障碍物中心的距离
        distances = np.linalg.norm(trajectory[:, :3] - obstacle_pos, axis=1)
        
        # 找出距离小于半径的点（碰撞点）
        collision_mask = distances < obstacle_radius
        collision_indices = np.where(collision_mask)[0]
        
        return collision_indices, distances, len(collision_indices) > 0
    
    # 创建两个子图
    fig = plt.figure(figsize=(16, 6))
    
    # === 无辅助轨迹图 ===
    ax1 = fig.add_subplot(121, projection='3d')
    
    trajectory_no_assist = results['no_assist']['trajectory']
    obstacle_pos = results['no_assist']['obstacle_pos']
    
    # 检测无辅助轨迹的碰撞
    collision_indices_no_assist, distances_no_assist, has_collision_no_assist = detect_collisions(
        trajectory_no_assist, obstacle_pos, obstacle_radius=1.0
    )
    
    # 绘制轨迹 - 根据是否碰撞使用不同颜色
    if has_collision_no_assist:
        # 如果有碰撞，分段绘制：安全部分和碰撞部分
        safe_mask = distances_no_assist >= 1.0
        
        # 绘制安全部分（绿色到橙色渐变表示接近程度）
        for i in range(len(trajectory_no_assist) - 1):
            if safe_mask[i] and safe_mask[i+1]:  # 两个点都安全
                # 根据距离设置颜色深浅
                danger_level = max(0, (2.0 - distances_no_assist[i]) / 1.0)  # 距离2m以内开始变色
                color_intensity = min(danger_level, 1.0)
                color = plt.cm.RdYlGn_r(color_intensity * 0.7)  # 使用红黄绿色谱的反向
                
                ax1.plot(trajectory_no_assist[i:i+2, 0], 
                        trajectory_no_assist[i:i+2, 1], 
                        trajectory_no_assist[i:i+2, 2], 
                        color=color, linewidth=3, alpha=0.8)
        
        if len(collision_indices_no_assist) > 0:
            collision_points = trajectory_no_assist[collision_indices_no_assist]
            ax1.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2],
                       c='yellow', s=80, marker='o', label='Collision Points', linewidth=2, alpha=0.9, zorder=10)
            
            middle_collision_idx = collision_indices_no_assist[len(collision_indices_no_assist) // 2]
            middle_point = trajectory_no_assist[middle_collision_idx]
            ax1.text(middle_point[0], middle_point[1], middle_point[2] + 1, '!', 
                    fontsize=30, fontweight='bold', color='red',
                    ha='center', va='center', zorder=15)
                
            # 连接碰撞点形成碰撞轨迹段
            for i in range(len(collision_indices_no_assist) - 1):
                if collision_indices_no_assist[i+1] - collision_indices_no_assist[i] == 1:  # 连续的碰撞点
                    idx1, idx2 = collision_indices_no_assist[i], collision_indices_no_assist[i+1]
                    ax1.plot(trajectory_no_assist[idx1:idx2+1, 0],
                            trajectory_no_assist[idx1:idx2+1, 1], 
                            trajectory_no_assist[idx1:idx2+1, 2],
                            color='yellow', linewidth=4, alpha=0.9)
    else:
        # 没有碰撞，正常绘制
        ax1.plot(trajectory_no_assist[:, 0], trajectory_no_assist[:, 1], trajectory_no_assist[:, 2], 
                color='#FF6B6B', linewidth=3, label='No Assistance', alpha=0.9)
    
    # 起点和终点
    ax1.scatter(trajectory_no_assist[0, 0], trajectory_no_assist[0, 1], trajectory_no_assist[0, 2], 
               color='green', s=100, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax1.scatter(trajectory_no_assist[-1, 0], trajectory_no_assist[-1, 1], trajectory_no_assist[-1, 2], 
               color='red', s=100, marker='X', label='End', edgecolors='black', linewidth=2)

    # 绘制障碍物
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = 1.0 * np.outer(np.cos(u), np.sin(v)) + obstacle_pos[0]
    y_sphere = 1.0 * np.outer(np.sin(u), np.sin(v)) + obstacle_pos[1]
    z_sphere = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle_pos[2]
    
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='#DC143C')
    
    # 添加碰撞统计到标题
    title1 = 'Without Assistance'
    ax1.set_title(title1, fontsize=24, fontweight='bold', y=0.95)
    
    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold',labelpad=0)
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold',labelpad=10)
    ax1.set_zlabel('Z (m)', fontsize=12, fontweight='bold',labelpad=0)
    # ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === 有辅助轨迹图 ===
    ax2 = fig.add_subplot(122, projection='3d')
    
    trajectory_with_assist = results['with_assist']['trajectory']
    obstacle_pos = results['with_assist']['obstacle_pos']
    
    # 检测有辅助轨迹的碰撞
    collision_indices_with_assist, distances_with_assist, has_collision_with_assist = detect_collisions(
        trajectory_with_assist, obstacle_pos, obstacle_radius=1.0
    )
    
    # 绘制轨迹 - 根据是否碰撞使用不同颜色
    if has_collision_with_assist:
        # 如果有碰撞，分段绘制
        safe_mask = distances_with_assist >= 1.0
        
        # 绘制安全部分
        for i in range(len(trajectory_with_assist) - 1):
            if safe_mask[i] and safe_mask[i+1]:
                danger_level = max(0, (2.0 - distances_with_assist[i]) / 1.0)
                color_intensity = min(danger_level, 1.0)
                color = plt.cm.RdYlGn_r(color_intensity * 0.7)
                
                ax2.plot(trajectory_with_assist[i:i+2, 0], 
                        trajectory_with_assist[i:i+2, 1], 
                        trajectory_with_assist[i:i+2, 2], 
                        color=color, linewidth=3, alpha=0.8)
        
        # 绘制碰撞部分
        if len(collision_indices_with_assist) > 0:
            collision_points = trajectory_with_assist[collision_indices_with_assist]
            ax2.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2],
                       c='red', s=80, marker='X', label='Collision Points', 
                       edgecolors='darkred', linewidth=2, alpha=0.9, zorder=10)
            
            # 连接碰撞点
            for i in range(len(collision_indices_with_assist) - 1):
                if collision_indices_with_assist[i+1] - collision_indices_with_assist[i] == 1:
                    idx1, idx2 = collision_indices_with_assist[i], collision_indices_with_assist[i+1]
                    ax2.plot(trajectory_with_assist[idx1:idx2+1, 0],
                            trajectory_with_assist[idx1:idx2+1, 1], 
                            trajectory_with_assist[idx1:idx2+1, 2],
                            color='red', linewidth=4, alpha=0.9)
    else:
        # 没有碰撞，正常绘制
        ax2.plot(trajectory_with_assist[:, 0], trajectory_with_assist[:, 1], trajectory_with_assist[:, 2], 
                color="#0A6BDA", linewidth=3, label='With Assistance', alpha=0.9)
    
    # 起点和终点
    ax2.scatter(trajectory_with_assist[0, 0], trajectory_with_assist[0, 1], trajectory_with_assist[0, 2], 
               color='green', s=100, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax2.scatter(trajectory_with_assist[-1, 0], trajectory_with_assist[-1, 1], trajectory_with_assist[-1, 2], 
               color='red', s=100, marker='X' \
               '', label='End', edgecolors='black', linewidth=2)
    
    # 绘制障碍物
    ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='#DC143C')
    
    # 添加碰撞统计到标题
    title2 = 'With Assistance'
    ax2.set_title(title2, fontsize=24, fontweight='bold', y=0.95)

    ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold',labelpad=0)
    ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold',labelpad=10)
    ax2.set_zlabel('Z (m)', fontsize=12, fontweight='bold',labelpad=0)
    # ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 设置相同的坐标轴范围以便比较
    all_x = np.concatenate([trajectory_no_assist[:, 0], trajectory_with_assist[:, 0], 
                           [obstacle_pos[0] - 1, obstacle_pos[0] + 1]])
    all_y = np.concatenate([trajectory_no_assist[:, 1], trajectory_with_assist[:, 1], 
                           [obstacle_pos[1] - 1, obstacle_pos[1] + 1]])
    all_z = np.concatenate([trajectory_no_assist[:, 2], trajectory_with_assist[:, 2], 
                           [obstacle_pos[2] - 1, obstacle_pos[2] + 1]])

    x_range = [all_x.min() - 0.5, all_x.max() + 0.5]
    y_range = [all_y.min() - 0.5, all_y.max() + 0.5]
    z_range = [all_z.min() - 0.5, all_z.max() + 0.5]
    
    # 计算Z轴的1米间隔刻度
    z_min_tick = np.floor(z_range[0])  
    z_max_tick = np.ceil(z_range[1])   
    z_ticks = np.arange(z_min_tick, z_max_tick + 1, 1.0)  
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # 设置Z轴刻度为1米间隔
        ax.set_zticks(z_ticks)
        
        ax.view_init(elev=30, azim=-30)
        x_display_range = 2 * (all_x.max()-all_x.min())  
        y_display_range = 2 * (all_y.max()-all_y.min())  
        z_display_range = 6 

        # 设置box aspect使显示比例与数据比例一致
        ax.set_box_aspect([x_display_range, y_display_range, z_display_range])
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    # ===== 第一步：数据收集 =====
    # print("开始数据收集...")
    # results = run_comparison_simulation(max_steps=5000)
    # save_simulation_results(results, "simulation_results.pkl")
    
    # ===== 第二步：读取数据并绘图 =====
    # 注释掉上面的数据收集部分，只运行下面的绘图部分
    results = load_simulation_results("simulation_results.pkl")
    plot_separate_trajectories(results)