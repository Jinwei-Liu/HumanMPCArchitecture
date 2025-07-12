import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from Personalized_SA.human_model.sac import SAC_countinuous
from Personalized_SA.env.quadrotor import *
from Personalized_SA.config.config import args
from Personalized_SA.human_model.rlhuman import scale_to_env

class PredictionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class HumanBehaviorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, predict_steps=50):
        super(HumanBehaviorPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_steps = predict_steps
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # Output layers for predicting future states
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 10)  # 10-dimensional state
            ) for _ in range(predict_steps)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for prediction
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        predictions = []
        for i in range(self.predict_steps):
            pred = self.output_layers[i](last_hidden)
            predictions.append(pred)
        
        # Stack predictions: (batch_size, predict_steps, 10)
        return torch.stack(predictions, dim=1)

class RLHuman:
    def __init__(self, state_dim: int, action_dim: int):
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

    def select_action(self, state: np.ndarray, deterministic: bool = False, temperature: float = 1.0) -> np.ndarray:
        action = self.agent.select_action(state, deterministic=deterministic, temperature=temperature)
        return action

def collect_episode_data(env, rlhuman):
    """收集一个episode的数据"""
    step_idx = 0
    done = False
    obs_dict, _ = env.reset()
    state = obs_dict["human"]
    
    states_history = []
    actions_history = []
    
    while not done and step_idx < args.max_steps:
        # 记录当前状态
        states_history.append(state[:10].copy())
        
        # 选择动作
        a_norm = rlhuman.select_action(state, deterministic=False, temperature=1)
        env_act = scale_to_env(a_norm, env.action_space["low"], env.action_space["high"])
        actions_history.append(env_act.copy())
        
        # 执行动作
        obs_dict, reward, done, info = env.step(env_act)
        next_state = obs_dict["human"]
        
        state = next_state
        step_idx += 1
    
    return np.array(states_history), np.array(actions_history)

def create_sequences(states, actions, sequence_length=3, predict_steps=50):
    """创建用于训练的序列数据"""
    sequences = []
    targets = []
    
    for i in range(len(states) - sequence_length - predict_steps + 1):
        # 输入序列：前3步的状态和动作
        seq_states = states[i:i+sequence_length]
        seq_actions = actions[i:i+sequence_length]
        
        # 将状态和动作连接作为输入特征
        seq_input = np.concatenate([seq_states, seq_actions], axis=1)
        sequences.append(seq_input)
        
        # 目标：后续50步的状态
        target_states = states[i+sequence_length:i+sequence_length+predict_steps]
        targets.append(target_states)
    
    return np.array(sequences), np.array(targets)

def combine_multiple_episodes_data(episodes_data, sequence_length=3, predict_steps=50):
    """合并多个episode的数据创建训练序列"""
    all_sequences = []
    all_targets = []
    
    for states, actions in episodes_data:
        sequences, targets = create_sequences(states, actions, sequence_length, predict_steps)
        all_sequences.append(sequences)
        all_targets.append(targets)
    
    # 合并所有episode的数据
    combined_sequences = np.concatenate(all_sequences, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)
    
    return combined_sequences, combined_targets

def train_model(model, train_loader, device, epochs=100):
    """训练神经网络模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return train_losses

def evaluate_model(model, test_data, device):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in test_data:
            sequences = torch.FloatTensor(sequences).unsqueeze(0).to(device)
            targets_np = targets
            
            predictions = model(sequences)
            predictions_np = predictions.cpu().numpy().squeeze()
            
            all_predictions.append(predictions_np)
            all_targets.append(targets_np)
    
    return all_predictions, all_targets

def calculate_state_wise_errors(predictions, targets, predict_steps_list=[5, 10, 20, 30, 40, 50]):
    """计算每个状态维度在不同预测步数下的误差"""
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']
    num_states = 10
    
    results = {}
    
    for predict_steps in predict_steps_list:
        # 初始化存储每个状态维度的误差
        state_errors = {state_name: [] for state_name in state_names}
        
        for pred, target in zip(predictions, targets):
            if len(target) >= predict_steps:
                pred_truncated = pred[:predict_steps]  # shape: (predict_steps, 10)
                target_truncated = target[:predict_steps]  # shape: (predict_steps, 10)
                
                # 计算每个状态维度的误差
                for state_idx in range(num_states):
                    state_name = state_names[state_idx]
                    
                    # 计算该状态维度在所有预测步数上的平均绝对误差
                    state_pred = pred_truncated[:, state_idx]
                    state_target = target_truncated[:, state_idx]
                    
                    # 计算每个时刻的绝对误差
                    abs_errors = np.abs(state_pred - state_target)
                    
                    # 存储该序列中该状态的平均误差
                    state_errors[state_name].append(np.mean(abs_errors))
        
        # 计算每个状态的统计信息
        step_results = {}
        for state_name in state_names:
            if state_errors[state_name]:
                errors = np.array(state_errors[state_name])
                step_results[state_name] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors)
                }
            else:
                step_results[state_name] = {
                    'mean_error': 0.0,
                    'std_error': 0.0,
                    'min_error': 0.0,
                    'max_error': 0.0
                }
        
        results[predict_steps] = step_results
    
    return results

def print_detailed_results(results):
    """打印详细的结果"""
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']

    print("\n" + "="*80)
    print("DETAILED STATE-WISE PREDICTION ERRORS")
    print("="*80)
    
    for predict_steps in sorted(results.keys()):
        print(f"\n{predict_steps} Steps Prediction:")
        print("-" * 60)
        print(f"{'State':<12} {'Mean Error':<12} {'Std Error':<12} {'Min Error':<12} {'Max Error':<12}")
        print("-" * 60)
        
        for state_name in state_names:
            metrics = results[predict_steps][state_name]
            print(f"{state_name:<12} {metrics['mean_error']:<12.6f} {metrics['std_error']:<12.6f} "
                  f"{metrics['min_error']:<12.6f} {metrics['max_error']:<12.6f}")

def create_error_summary_table(results):
    """创建误差汇总表"""
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']
    predict_steps_list = sorted(results.keys())
    
    # 创建DataFrame用于更好的可视化
    summary_data = []
    
    for state_name in state_names:
        row = {'State': state_name}
        for steps in predict_steps_list:
            mean_error = results[steps][state_name]['mean_error']
            std_error = results[steps][state_name]['std_error']
            row[f'{steps}_steps_mean'] = mean_error
            row[f'{steps}_steps_std'] = std_error
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df

def plot_error_heatmap(results, save_path="error_heatmap.png"):
    """绘制误差热力图"""
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']
    predict_steps_list = sorted(results.keys())
    
    # 创建误差矩阵
    error_matrix = np.zeros((len(state_names), len(predict_steps_list)))
    
    for i, state_name in enumerate(state_names):
        for j, steps in enumerate(predict_steps_list):
            error_matrix[i, j] = results[steps][state_name]['mean_error']
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(error_matrix, cmap='viridis', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(predict_steps_list)))
    ax.set_xticklabels([f'{steps} steps' for steps in predict_steps_list])
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels(state_names)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Mean Absolute Error', rotation=270, labelpad=20)
    
    # 添加数值标注
    for i in range(len(state_names)):
        for j in range(len(predict_steps_list)):
            text = ax.text(j, i, f'{error_matrix[i, j]:.4f}',
                         ha="center", va="center", color="white", fontsize=8)
    
    plt.title('State-wise Prediction Errors Across Different Prediction Horizons')
    plt.xlabel('Prediction Steps')
    plt.ylabel('State Dimensions')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_trends(results, save_path="error_trends.png"):
    """绘制误差趋势图"""
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']
    predict_steps_list = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, state_name in enumerate(state_names):
        means = [results[steps][state_name]['mean_error'] for steps in predict_steps_list]
        stds = [results[steps][state_name]['std_error'] for steps in predict_steps_list]
        
        axes[i].errorbar(predict_steps_list, means, yerr=stds, 
                        marker='o', capsize=5, capthick=2, linewidth=2)
        axes[i].set_title(f'{state_name} Prediction Error')
        axes[i].set_xlabel('Prediction Steps')
        axes[i].set_ylabel('Mean Absolute Error')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')  # 使用对数刻度以便更好地显示不同量级的误差
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 环境设置
    env = QuadrotorRaceEnv(dt=0.01)
    action_low = env.action_space["low"]
    action_high = env.action_space["high"]
    state_dim = env.observation_dim_human
    action_dim = action_low.shape[0]
    
    # 初始化RL智能体
    rlhuman = RLHuman(state_dim, action_dim)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 超参数设置
    num_episodes = 10
    num_train_episodes = 5  # 新增超参数：用于训练的episode数量
    
    # 收集多轮数据
    print("Collecting episode data...")
    all_episodes_data = []
    
    for episode in range(num_episodes):
        print(f"Collecting episode {episode + 1}/{num_episodes}")
        states, actions = collect_episode_data(env, rlhuman)
        all_episodes_data.append((states, actions))
        print(f"Episode {episode + 1}: {len(states)} steps")
    
    # 数据划分：前num_train_episodes个episode用于训练，其余用于测试
    train_episodes_idx = list(range(num_train_episodes))
    test_episodes_idx = list(range(num_train_episodes, num_episodes))
    
    print(f"\nUsing episodes {[i+1 for i in train_episodes_idx]} for training")
    print(f"Using episodes {[i+1 for i in test_episodes_idx]} for testing")
    
    # 准备训练数据：合并多个训练episode的数据
    train_episodes_data = [all_episodes_data[i] for i in train_episodes_idx]
    train_sequences, train_targets = combine_multiple_episodes_data(
        train_episodes_data, sequence_length=3, predict_steps=50
    )
    
    print(f"Training data: {len(train_sequences)} sequences from {num_train_episodes} episodes")
    
    # 创建数据加载器
    train_dataset = PredictionDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    input_dim = train_sequences.shape[2]  # 状态维度 + 动作维度
    model = HumanBehaviorPredictor(input_dim=input_dim, hidden_dim=128, 
                                  num_layers=2, predict_steps=50).to(device)
    
    print(f"Model input dimension: {input_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("\nTraining model...")
    train_losses = train_model(model, train_loader, device, epochs=100)
    
    # 准备测试数据
    test_data = []
    for test_idx in test_episodes_idx:
        test_states, test_actions = all_episodes_data[test_idx]
        test_sequences, test_targets = create_sequences(test_states, test_actions, 
                                                       sequence_length=3, predict_steps=50)
        
        for i in range(len(test_sequences)):
            test_data.append((test_sequences[i], test_targets[i]))
    
    print(f"Test data: {len(test_data)} sequences from {len(test_episodes_idx)} episodes")
    
    # 评估模型
    print("\nEvaluating model...")
    predictions, targets = evaluate_model(model, test_data, device)
    
    # 计算每个状态维度的误差
    state_wise_results = calculate_state_wise_errors(predictions, targets, 
                                                    predict_steps_list=[5, 10, 20, 30, 40, 50])
    
    # 打印详细结果
    print_detailed_results(state_wise_results)
    
    # 创建汇总表
    summary_df = create_error_summary_table(state_wise_results)
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # 保存汇总表
    summary_df.to_csv(f'Personalized_SA/nn_prediction/state_wise_prediction_errors_train{args.threshold_vel}.csv', index=False)
    print(f"\nSummary table saved as 'state_wise_prediction_errors_train{args.threshold_vel}.csv'")

    # 绘制可视化结果
    plot_error_heatmap(state_wise_results, f"Personalized_SA/nn_prediction/state_wise_error_heatmap_train{args.threshold_vel}.png")
    plot_error_trends(state_wise_results, f"Personalized_SA/nn_prediction/state_wise_error_trends_train{args.threshold_vel}.png")

    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title(f'Training Loss (Using {num_train_episodes} Episodes)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(f"Personalized_SA/nn_prediction/training_loss_train{args.threshold_vel}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), f'Personalized_SA/nn_prediction/human_behavior_predictor_train{args.threshold_vel}.pth')
    print(f"\nModel saved as 'human_behavior_predictor_train{args.threshold_vel}.pth'")

if __name__ == "__main__":
    main()
