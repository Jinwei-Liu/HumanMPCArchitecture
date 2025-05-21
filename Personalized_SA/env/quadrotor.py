import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch, torch.nn as nn
#
from Personalized_SA.env.quad_index import *

class Quadrotor_v0(object):
    #
    def __init__(self, dt):
        self.s_dim = 10
        self.a_dim = 4
        #
        self._state = np.zeros(shape=self.s_dim)
        self._state[kQuatW] = 1.0
        #
        self._actions = np.zeros(shape=self.a_dim)

        #
        self._gz = 9.81
        self._dt = dt
        self._arm_l = 0.3   # m
        
        # Sampling range of the quadrotor's initial position
        self._xyz_dist = np.array(
            [ [-3.0, -1.0], # x 
              [-2.0, 2.],   # y
              [0.0, 2.5]]   # z
        )
        # Sampling range of the quadrotor's initial velocity
        self._vxyz_dist = np.array(
            [ [-1.0, 1.0],  # vx
              [-1.0, 1.0],  # vy
              [-1.0, 1.0]]  # vz
        )
        
        # x, y, z, r, p, y, vx, vy, vz
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])
        #
        self.reset()
        # self._t = 0.0
    
    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        self._state[kQuatW] = 1.0 # 
        #
        # initialize position, randomly
        self._state[kPosX] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1])
        self._state[kPosY] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1])
        self._state[kPosZ] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1])
        
        # initialize rotation, randomly
        quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
        # normalize the quaternion
        self._state[kQuatW:kQuatZ+1] = quad_quat0 / np.linalg.norm(quad_quat0)
        
        # initialize velocity, randomly
        self._state[kVelX] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1])
        self._state[kVelY] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1])
        self._state[kVelZ] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1])
        #
        return self._state

    def run(self, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        """
        # rk4 int
        M = 10
        DT = self._dt / M
        #
        X = self._state
        for i in range(M):
            k1 = DT*self._f(X, action)
            k2 = DT*self._f(X + 0.5*k1, action)
            k3 = DT*self._f(X + 0.5*k2, action)
            k4 = DT*self._f(X + k3, action)
            #
            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
        #
        self._state = X
        self.set_quaternion()
    
        return self._state

    def _f(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        thrust, wx, wy, wz = action
        #
        dstate = np.zeros(shape=self.s_dim)

        dstate[kPosX:kPosZ+1] = state[kVelX:kVelZ+1]

        qw, qx, qy, qz = self.get_quaternion()

        dstate[kQuatW] = 0.5 * ( -wx*qx - wy*qy - wz*qz )
        dstate[kQuatX] = 0.5 * (  wx*qw + wz*qy - wy*qz )
        dstate[kQuatY] = 0.5 * (  wy*qw - wz*qx + wx*qz )
        dstate[kQuatZ] = 0.5 * (  wz*qw + wy*qx - wx*qy )

        dstate[kVelX] = 2 * ( qw*qy + qx*qz ) * thrust
        dstate[kVelY] = 2 * ( qy*qz - qw*qx ) * thrust
        dstate[kVelZ] = (qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz

        return dstate

    def set_state(self, state):
        """
        Set the vehicle's state
        """
        self._state = state
        
    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_cartesian_state(self):
        """
        Get the Full state in Cartesian coordinates
        """
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()
        return cartesian_state
    
    def get_position(self,):
        """
        Retrieve Position
        """
        return self._state[kPosX:kPosZ+1]
    
    def get_velocity(self,):
        """
        Retrieve Linear Velocity
        """
        return self._state[kVelX:kVelZ+1]
    
    def set_quaternion(self,):
        """
        Set Quaternion
        """
        quat = np.zeros(4)
        quat = self._state[kQuatW:kQuatZ+1]
        quat = quat / np.linalg.norm(quat)
        self._state[kQuatW:kQuatZ+1] = quat

    def get_quaternion(self,):
        """
        Retrieve Quaternion
        """
        quat = np.zeros(4)
        quat = self._state[kQuatW:kQuatZ+1]
        quat = quat / np.linalg.norm(quat)
        return quat

    def get_euler(self,):
        """
        Retrieve Euler Angles of the Vehicle
        """
        quat = self.get_quaternion()
        euler = self._quatToEuler(quat)
        return euler

    def get_axes(self):
        """
        Get the 3 axes (x, y, z) in world frame (for visualization only)
        """
        # axes in body frame
        b_x = np.array([self._arm_l, 0, 0])
        b_y = np.array([0, self._arm_l, 0])
        b_z = np.array([0, 0,  -self._arm_l])
        
        # rotation matrix
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()
        
        # axes in body frame
        w_x = rot_matrix@b_x + quad_center
        w_y = rot_matrix@b_y + quad_center
        w_z = rot_matrix@b_z + quad_center
        return [w_x, w_y, w_z]

    def get_motor_pos(self):
        """
        Get the 4 motor poses in world frame (for visualization only)
        """
        # motor position in body frame
        b_motor1 = np.array([np.sqrt(self._arm_l/2), np.sqrt(self._arm_l/2), 0])
        b_motor2 = np.array([-np.sqrt(self._arm_l/2), np.sqrt(self._arm_l/2), 0])
        b_motor3 = np.array([-np.sqrt(self._arm_l/2), -np.sqrt(self._arm_l/2), 0])
        b_motor4 = np.array([np.sqrt(self._arm_l/2), -np.sqrt(self._arm_l/2), 0])
        #
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()
        
        # motor position in world frame
        w_motor1 = rot_matrix@b_motor1 + quad_center
        w_motor2 = rot_matrix@b_motor2 + quad_center
        w_motor3 = rot_matrix@b_motor3 + quad_center
        w_motor4 = rot_matrix@b_motor4 + quad_center
        return [w_motor1, w_motor2, w_motor3, w_motor4]

    @staticmethod
    def _quatToEuler(quat):
        """
        Convert Quaternion to Euler Angles
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
        euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
        euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
        return [euler_x, euler_y, euler_z]

#
class Quadrotor_MPC(nn.Module):
    def __init__(self, dt):
        super().__init__()
        self.s_dim = 10
        self.a_dim = 4

        self._gz = 9.81
        self._dt = dt

    def forward(self, X, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        """
        DT = self._dt

        k1 = DT*self._f(X, action)
        X = X + k1

        return X

    def _f(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        thrust, wx, wy, wz = action.split(1, dim=-1)

        dstate = torch.zeros_like(state)

        dstate[...,kPosX:kPosZ+1] = state[...,kVelX:kVelZ+1]

        quat = state[...,kQuatW:kQuatZ+1]
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)

        qw, qx, qy, qz = quat.split(1, dim=-1)

        dstate[...,kQuatW] = (0.5 * ( -wx*qx - wy*qy - wz*qz )).squeeze(-1)
        dstate[...,kQuatX] = (0.5 * (  wx*qw + wz*qy - wy*qz )).squeeze(-1)
        dstate[...,kQuatY] = (0.5 * (  wy*qw - wz*qx + wx*qz )).squeeze(-1)
        dstate[...,kQuatZ] = (0.5 * (  wz*qw + wy*qx - wx*qy )).squeeze(-1)

        dstate[...,kVelX] = (2 * ( qw*qy + qx*qz ) * thrust).squeeze(-1)
        dstate[...,kVelY] = (2 * ( qy*qz - qw*qx ) * thrust).squeeze(-1)
        dstate[...,kVelZ] = ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz).squeeze(-1)

        return dstate
    
    def grad_input(self, X: torch.Tensor, action: torch.Tensor):
        """
        Return A_d = d x_{k+1}/d x_k  and  B_d = d x_{k+1}/d u_k
        Shapes:
            A_d  [..., s_dim, s_dim]
            B_d  [..., s_dim, a_dim]
        """
        DT = self._dt
        batch_shape = X.shape[:-1]          # 允许任意批量 / 时间维

        # ---------- 拆分输入 ----------
        thrust, wx, wy, wz = action.split(1, dim=-1)   # (…,1)

        quat = X[..., kQuatW:kQuatZ+1]
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)  # 保证单位四元数
        qw, qx, qy, qz = quat.split(1, dim=-1)         # (…,1)

        # ---------- 连续时间 Jacobian ----------
        A = X.new_zeros(*batch_shape, self.s_dim, self.s_dim)
        B = X.new_zeros(*batch_shape, self.s_dim, self.a_dim)

        # ① 位置对速度
        A[..., kPosX, kVelX] = 1.0
        A[..., kPosY, kVelY] = 1.0
        A[..., kPosZ, kVelZ] = 1.0

        # ② 四元数运动学  0.5*q ⊗ [0, ω]
        A[..., kQuatW, kQuatX] = -0.5 * wx.squeeze(-1)
        A[..., kQuatW, kQuatY] = -0.5 * wy.squeeze(-1)
        A[..., kQuatW, kQuatZ] = -0.5 * wz.squeeze(-1)

        A[..., kQuatX, kQuatW] =  0.5 * wx.squeeze(-1)
        A[..., kQuatX, kQuatY] =  0.5 * wz.squeeze(-1)
        A[..., kQuatX, kQuatZ] = -0.5 * wy.squeeze(-1)

        A[..., kQuatY, kQuatW] =  0.5 * wy.squeeze(-1)
        A[..., kQuatY, kQuatX] = -0.5 * wz.squeeze(-1)
        A[..., kQuatY, kQuatZ] =  0.5 * wx.squeeze(-1)

        A[..., kQuatZ, kQuatW] =  0.5 * wz.squeeze(-1)
        A[..., kQuatZ, kQuatX] =  0.5 * wy.squeeze(-1)
        A[..., kQuatZ, kQuatY] = -0.5 * wx.squeeze(-1)

        # ③ 速度对姿态（由 R(q)·e3*T 产生）
        twoT = 2.0 * thrust.squeeze(-1)
        # vx 行
        A[..., kVelX, kQuatW] =  twoT * qy.squeeze(-1)
        A[..., kVelX, kQuatX] =  twoT * qz.squeeze(-1)
        A[..., kVelX, kQuatY] =  twoT * qw.squeeze(-1)
        A[..., kVelX, kQuatZ] =  twoT * qx.squeeze(-1)
        # vy 行
        A[..., kVelY, kQuatW] = -twoT * qx.squeeze(-1)
        A[..., kVelY, kQuatX] = -twoT * qw.squeeze(-1)
        A[..., kVelY, kQuatY] =  twoT * qz.squeeze(-1)
        A[..., kVelY, kQuatZ] =  twoT * qy.squeeze(-1)
        # vz 行
        A[..., kVelZ, kQuatW] =  2.0 * thrust.squeeze(-1) * qw.squeeze(-1)
        A[..., kVelZ, kQuatX] = -2.0 * thrust.squeeze(-1) * qx.squeeze(-1)
        A[..., kVelZ, kQuatY] = -2.0 * thrust.squeeze(-1) * qy.squeeze(-1)
        A[..., kVelZ, kQuatZ] =  2.0 * thrust.squeeze(-1) * qz.squeeze(-1)
        
        # ④ 连续时间 B
        #    四元数对机体系角速度
        B[..., kQuatW, 1] = -0.5 * qx.squeeze(-1)
        B[..., kQuatW, 2] = -0.5 * qy.squeeze(-1)
        B[..., kQuatW, 3] = -0.5 * qz.squeeze(-1)

        B[..., kQuatX, 1] =  0.5 * qw.squeeze(-1)
        B[..., kQuatX, 2] = -0.5 * qz.squeeze(-1)
        B[..., kQuatX, 3] =  0.5 * qy.squeeze(-1)

        B[..., kQuatY, 1] =  0.5 * qz.squeeze(-1)
        B[..., kQuatY, 2] =  0.5 * qw.squeeze(-1)
        B[..., kQuatY, 3] = -0.5 * qx.squeeze(-1)

        B[..., kQuatZ, 1] = -0.5 * qy.squeeze(-1)
        B[..., kQuatZ, 2] =  0.5 * qx.squeeze(-1)
        B[..., kQuatZ, 3] =  0.5 * qw.squeeze(-1)

        #    速度对推力
        B[..., kVelX, 0] =  2.0 * (qw*qy + qx*qz).squeeze(-1)
        B[..., kVelY, 0] = -2.0 * (qw*qx - qy*qz).squeeze(-1)
        B[..., kVelZ, 0] =  (qw*qw - qx*qx - qy*qy + qz*qz).squeeze(-1)

        # ---------- 离散化 ----------
        eye = torch.eye(self.s_dim, dtype=X.dtype, device=X.device)
        eye = eye.expand(*batch_shape, -1, -1)          # 广播到批量
        A_d = eye + DT * A
        B_d = DT * B
        return A_d, B_d
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods
import matplotlib.pyplot as plt
def test_mpc():
    # ------------ 关键超参数 ------------
    DT          = 0.02          # 积分步长  (s)
    T_HORIZON   = 50         # MPC 预测步数
    step = 500

    # ---------- 初始化 ----------
    quad = Quadrotor_MPC(DT)
    quad_env = Quadrotor_v0(DT)
    quad_env.reset()

    n_state, n_ctrl   = quad.s_dim, quad.a_dim

    # ----------- 1. 目标状态 & 权重 -----------------
    x_goal            = torch.zeros(n_state)
    x_goal[kQuatW]    = 1.0                       # 悬停姿态
    x_goal[kPosX]    = 10.0 

    w_pos, w_vel      = 1., 0.001
    w_quat            = 0.001
    w_act             = 0.00001
    n_batch           = 1

    # ----------- 2. 打包成 C, c --------------------
    goal_weights = torch.Tensor([w_pos, w_pos, w_pos,              # 位置
         w_quat, w_quat, w_quat, w_quat,   # 四元数 (w,x,y,z)
         w_vel, w_vel, w_vel]            # 速度
            )
    ctrl_weights = torch.Tensor([w_act,w_act,w_act,w_act])

    q = torch.cat((
    goal_weights,
    ctrl_weights
    ))

    px = -torch.sqrt(goal_weights)*x_goal

    p = torch.cat((px, torch.zeros(n_ctrl)))

    C = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1, 1)
    c = p.unsqueeze(0).repeat(T_HORIZON, n_batch, 1)

    cost = QuadCost(C, c)                         

    # ----------- 3. 控制器 -------------------------
    u_min = torch.tensor([0.0, -2.0, -2.0, -2.0])
    u_max = torch.tensor([20.0,  2.0,  2.0,  2.0])

    u_lower = u_min.repeat(T_HORIZON, 1, 1)   # (25, 4)
    u_upper = u_max.repeat(T_HORIZON, 1, 1)   # (25, 4)

    ctrl = mpc.MPC(n_state=n_state,
                   n_ctrl=n_ctrl,
                   T=T_HORIZON,
                   u_lower=u_lower,
                   u_upper=u_upper,
                   lqr_iter=10,
                   grad_method=GradMethods.AUTO_DIFF,
                   exit_unconverged = False,
                   eps=1e-2,
                   verbose=0)

    # ----------- 4. 主循环 -------------------------
    x = torch.zeros(n_state)
    x[kQuatW] = 1.0
    x[kPosZ] = -1.0
    x = x.unsqueeze(0) 
    quad_env.set_state(x.squeeze(0).detach().cpu().numpy())

    x_history = []
    for i in range(step):
        print(i)
        _, u_opt, _ = ctrl(x, cost, quad)
        state = quad_env.run(u_opt[0,0].detach().cpu().numpy())
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        x_history.append(state)

    x_arr = np.stack(x_history)
    step, n_state = x_arr.shape
    # === 2. 创建子图 =====================
    fig, axes = plt.subplots(n_state,      # 行数 = 状态维度
                            1,            # 一列
                            sharex=True,  # 共用 x 轴
                            figsize=(6, 1.8*n_state))

    for dim in range(n_state):
        ax = axes[dim]                    # 当前子图
        ax.plot(x_arr[:, dim])            # 画出 dim 维
        ax.set_ylabel(f"x[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Timestep")       # 只在最后一行标注
    fig.suptitle("Evolution of state vector x", y=1.02)
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_mpc()