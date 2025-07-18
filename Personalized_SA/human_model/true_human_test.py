import numpy as np
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Personalized_SA.nn_prediction.nn_prediction import *

# Directory where data is saved
SAVE_DIR = "./Personalized_SA/human_model/true_human"

def get_available_state_files():
    """
    List all states_*.npy files in SAVE_DIR, sorted by timestamp (from filename).
    Returns a list of timestamps (e.g., ['20241001-120000', '20241001-120001']).
    """
    files = [f for f in os.listdir(SAVE_DIR) if f.startswith("states_") and f.endswith(".npy")]
    # Extract timestamps and sort them (assuming format states_YYYYMMDD-HHMMSS.npy)
    timestamps = sorted([f[len("states_"):-4] for f in files])
    return timestamps

def read_and_display_states(timestamp=None):
    """
    Load states from a specific .npy file (or the latest if timestamp is None)
    and display x, y, z one group (step) at a time.
    
    Assumes each state is a 1D array with x, y, z as the first three elements.
    Adjust the indices based on the actual structure of obs["human"].
    """
    timestamps = get_available_state_files()
    
    if not timestamps:
        print(f"No states_*.npy files found in {SAVE_DIR}")
        return
    
    if timestamp is None:
        # Use the latest timestamp by default
        timestamp = timestamps[-1]
        print(f"No timestamp provided. Using the latest: {timestamp}")
    elif timestamp not in timestamps:
        print(f"Timestamp {timestamp} not found. Available timestamps: {timestamps}")
        return
    
    states_path = os.path.join(SAVE_DIR, f"states_{timestamp}.npy")
    states_arr = np.load(states_path)
    print(f"Loaded states from {states_path} with shape: {states_arr.shape}")
    
    # Extract x, y, z for all steps (assuming first three dimensions)
    x = states_arr[:, 0]  # Adjust index if x is not at position 0
    y = states_arr[:, 1]  # Adjust index if y is not at position 1
    z = states_arr[:, 2]  # Adjust index if z is not at position 2
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o', linestyle='-', color='b', label='Trajectory')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Trajectory from states_{timestamp}.npy')
    ax.legend()
    
    # Show the plot
    plt.show()

def read_states(timestamp=None):
    """
    Load states from a specific .npy file (or the latest if timestamp is None)
    and display x, y, z one group (step) at a time.
    
    Assumes each state is a 1D array with x, y, z as the first three elements.
    Adjust the indices based on the actual structure of obs["human"].
    """
    timestamps = get_available_state_files()
    
    if not timestamps:
        print(f"No states_*.npy files found in {SAVE_DIR}")
        return
    
    if timestamp is None:
        # Use the latest timestamp by default
        timestamp = timestamps[-1]
        print(f"No timestamp provided. Using the latest: {timestamp}")
    elif timestamp not in timestamps:
        print(f"Timestamp {timestamp} not found. Available timestamps: {timestamps}")
        return
    
    states_path = os.path.join(SAVE_DIR, f"states_{timestamp}.npy")
    states_arr = np.load(states_path)

    actions_path = os.path.join(SAVE_DIR, f"actions_{timestamp}.npy")
    actions_arr = np.load(actions_path)

    return states_arr[:,:10], actions_arr


def create_error_summary_table_h(results):
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
            max_error = results[steps][state_name]['max_error']
            row[f'{steps}_steps_mean'] = mean_error
            row[f'{steps}_steps_std'] = std_error
            row[f'{steps}_steps_max'] = max_error
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df

def nn_prediction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Alternatively, prompt user to select from list
    timestamps = get_available_state_files()
    
    # for selected_timestamp in timestamps:
    #     read_and_display_states(selected_timestamp)

    num_train_episodes = 3
    num_episodes = len(timestamps)
    all_episodes_data = []
    
    for selected_timestamp in timestamps:
        states, actions = read_states(selected_timestamp)
        print(f"Loaded states and actions from {selected_timestamp} with shapes: {states.shape}, {actions.shape}")
        all_episodes_data.append((states, actions))
    
    # 数据划分：前num_train_episodes个episode用于训练，其余用于测试
    train_episodes_idx = list(range(num_train_episodes))
    test_episodes_idx = list(range(num_train_episodes, num_episodes))
    
    print(f"\nUsing episodes {[i+1 for i in train_episodes_idx]} for training")
    print(f"Using episodes {[i+1 for i in test_episodes_idx]} for testing")
    
    # 准备训练数据：合并多个训练episode的数据
    train_episodes_data = [all_episodes_data[i] for i in train_episodes_idx]
    train_sequences, train_targets = combine_multiple_episodes_data(
        train_episodes_data, sequence_length=3, predict_steps=100
    )
    
    print(f"Training data: {len(train_sequences)} sequences from {num_train_episodes} episodes")
    
    # 创建数据加载器
    train_dataset = PredictionDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    input_dim = train_sequences.shape[2]  # 状态维度 + 动作维度
    model = HumanBehaviorPredictor(input_dim=input_dim, hidden_dim=128, 
                                  num_layers=2, predict_steps=100).to(device)
    
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
                                                       sequence_length=3, predict_steps=100)
        
        for i in range(len(test_sequences)):
            test_data.append((test_sequences[i], test_targets[i]))
    
    print(f"Test data: {len(test_data)} sequences from {len(test_episodes_idx)} episodes")
    
    # 评估模型
    print("\nEvaluating model...")
    predictions, targets = evaluate_model(model, test_data, device)
    
    # 计算每个状态维度的误差
    state_wise_results = calculate_state_wise_errors(predictions, targets, 
                                                    predict_steps_list=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # 打印详细结果
    print_detailed_results(state_wise_results)
    # 创建汇总表
    summary_df = create_error_summary_table_h(state_wise_results)
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # 保存汇总表
    nn_csv_path = 'Personalized_SA/visualization/true_human.csv'
    summary_df.to_csv(nn_csv_path, index=False)
    print(f"\nSummary table saved as '{nn_csv_path}'")

    # 绘制可视化结果
    plot_error_heatmap(state_wise_results, f"Personalized_SA/nn_prediction/state_wise_error_heatmap_train_true_human.png")
    plot_error_trends(state_wise_results, f"Personalized_SA/nn_prediction/state_wise_error_trends_train_true_human.png")

    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title(f'Training Loss (Using {num_train_episodes} Episodes)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(f"Personalized_SA/nn_prediction/training_loss_train_true_human.png", dpi=300, bbox_inches='tight')
    plt.show()

def identify_human():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.optim as optim
    import torch.nn.utils as nn_utils

    # Load pre-recorded actions and states from a human model
    timestamps = get_available_state_files()
    states, actions = read_states(timestamps[0])
    actions = np.array(actions)
    states = np.array(states)

    x_arr = np.array(states)
    action_arr = np.array(actions)
    x_goal_arr = x_arr.copy()

    print("x_arr.shape:", x_arr.shape)
    print("action_arr.shape:", action_arr.shape)
    print("x_goal_arr.shape:", x_goal_arr.shape)

    DT = 0.01          # Integration step size (s)
    T_HORIZON = 50     # MPC prediction steps

    quad = Quadrotor_MPC(DT)
    n_state = quad.s_dim
    n_ctrl = quad.a_dim

    n_batch = x_arr.shape[0]     # Use all samples as a batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_all = torch.from_numpy(x_arr).float().to(device)
    action_all = torch.from_numpy(action_arr).float().to(device)
    x_goal_init = torch.cat((x_all, action_all), dim=1)

    # Define column indices that require shared parameters.
    # All values within each of these columns will use the same scalar parameter.
    shared_columns = []
    total_dims = n_state + n_ctrl

    # Create parameters for each column
    column_params = {}
    for i in range(total_dims):
        if i in shared_columns:
            # For columns with shared parameters, create a single scalar parameter.
            column_params[i] = torch.nn.Parameter(
                torch.tensor(x_goal_init[:, i].mean().item(), device=device),
                requires_grad=True
            )
        else:
            # For other columns, maintain the original per-row parameters.
            column_params[i] = torch.nn.Parameter(x_goal_init[:, i].clone(), requires_grad=True)

    def construct_x_goal_param():
        """
        Constructs the complete x_goal_param tensor.
        For specified columns, it uses the same scalar value for all rows.
        """
        x_goal_param = torch.zeros(n_batch, total_dims, device=device)
        
        for col in range(total_dims):
            if col in shared_columns:
                # For shared-parameter columns, broadcast the scalar value to the entire column.
                x_goal_param[:, col] = column_params[col]
            else:
                # For other columns, use independent parameter values for each row.
                x_goal_param[:, col] = column_params[col]
        
        return x_goal_param

    goal_weights = torch.ones(n_state, device=device) * 1e-2
    goal_weights[0:3] = 0.5       # Set larger initial values for position dimensions.
    goal_weights.requires_grad_(True)

    ctrl_weights = torch.ones(n_ctrl, device=device) * 1e-1
    ctrl_weights.requires_grad_(True)

    u_min = torch.tensor([0.0, -20.0, -20.0, -20.0], device=device)
    u_max = torch.tensor([100.0,  20.0,  20.0,  20.0], device=device)
    u_lower = u_min.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)
    u_upper = u_max.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1)

    optimizer_weights = optim.Adam([goal_weights, ctrl_weights], lr=0.01)
    optimizer_goal    = optim.Adam(list(column_params.values()), lr=0.01)

    for epoch in range(500):
        # Construct the complete x_goal_param tensor for the current iteration
        x_goal_param = construct_x_goal_param()
        
        q_vector = torch.cat((goal_weights**2, ctrl_weights**2), dim=0)
        Q_diag = torch.diag(q_vector)
        C = Q_diag.unsqueeze(0).unsqueeze(0).repeat(T_HORIZON, n_batch, 1, 1)

        px = -torch.sqrt(q_vector).unsqueeze(0) * x_goal_param
        p_all = px
        c = p_all.unsqueeze(0).repeat(T_HORIZON, 1, 1)

        x_batch = x_all
        cost = QuadCost(C, c)

        ctrl = mpc.MPC(
            n_state=n_state,
            n_ctrl=n_ctrl,
            T=T_HORIZON,
            u_lower=u_lower,
            u_upper=u_upper,
            lqr_iter=5,
            grad_method=GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
            verbose=0
        )
        _, u_opt, _ = ctrl(x_batch, cost, quad)

        u_pred = u_opt[0, :, :]
        delta_u = u_pred - action_all
        total_cost = torch.sum(delta_u**2)

        optimizer_weights.zero_grad()
        optimizer_goal.zero_grad()

        # Alternate between updating weights and goal parameters
        if epoch % 2 == 0:
            # Freeze goal parameters to update weights
            for param in column_params.values():
                param.requires_grad_(False)

            total_cost.backward()
            nn_utils.clip_grad_norm_([goal_weights, ctrl_weights], max_norm=1.0)
            optimizer_weights.step()

            # Unfreeze goal parameters for the next iteration
            for param in column_params.values():
                param.requires_grad_(True)

            print(f"Epoch {epoch:03d} | [Update weights] total_cost = {total_cost.item():.6f}")
            if goal_weights.grad is not None:
                print(f"  grad norm goal_weights = {goal_weights.grad.norm().item():.6f}")
            if ctrl_weights.grad is not None:
                print(f"  grad norm ctrl_weights = {ctrl_weights.grad.norm().item():.6f}")
            print(f"  current goal_weights = {goal_weights.data.cpu().numpy()}")
            print(f"  current ctrl_weights = {ctrl_weights.data.cpu().numpy()}")

        else:
            # Freeze weights to update goal parameters
            goal_weights.requires_grad_(False)
            ctrl_weights.requires_grad_(False)

            total_cost.backward()
            nn_utils.clip_grad_norm_(list(column_params.values()), max_norm=1.0)
            optimizer_goal.step()
            
            # Unfreeze weights for the next iteration
            goal_weights.requires_grad_(True)
            ctrl_weights.requires_grad_(True)

            print(f"Epoch {epoch:03d} | [Update x_goal_arr] total_cost = {total_cost.item():.6f}")
            
            # Print gradients and current values for shared parameter columns
            for col in shared_columns:
                if column_params[col].grad is not None:
                    print(f"  grad column {col} = {column_params[col].grad.item():.6f}")
                    print(f"  current column {col} value = {column_params[col].data.item():.6f}")
            
            # Print average gradient norm for other (non-shared) columns
            other_columns = [i for i in range(total_dims) if i not in shared_columns]
            other_grad_norms = []
            for col in other_columns:
                if column_params[col].grad is not None:
                    other_grad_norms.append(column_params[col].grad.norm().item())
            if other_grad_norms:
                avg_other_grad_norm = np.mean(other_grad_norms)
                print(f"  avg grad norm other columns = {avg_other_grad_norm:.6f}")

    print("Training finished!")

    # Get the final inferred goal state array
    x_goal_final = construct_x_goal_param().data.cpu().numpy()

    # Verify if the values within each shared column are identical
    print("\n=== Verification of Shared Columns ===")
    for col in shared_columns:
        print(f"\nColumn {col}:")
        print(f"  Scalar parameter value: {column_params[col].data.item():.6f}")
        print(f"  First 5 values in column: {x_goal_final[:5, col]}")
        is_uniform = np.allclose(x_goal_final[:, col], x_goal_final[0, col])
        print(f"  Is column {col} uniform? {is_uniform}")
        if is_uniform:
            print(f"  ✓ Column {col} successfully uses a shared parameter.")
        else:
            print(f"  ✗ Column {col} failed to use a shared parameter.")

    # Print the final shared parameter values
    print(f"\n=== Final Shared Parameter Values ===")
    for col in shared_columns:
        print(f"Column {col}: {column_params[col].data.item():.6f}")

    last_three_columns = x_goal_final[:, -3:]
    row_mean = np.mean(np.sqrt(np.sum(last_three_columns**2, axis=1)))
    print(f"\nMean norm of last three columns (angular velocities): {row_mean}")

    # Visualization
    x_state = x_arr
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_goal_final[:, 0], x_goal_final[:, 1], x_goal_final[:, 2], c='r', marker='o', label='Inferred Goal States')
    ax.scatter(x_state[:, 0], x_state[:, 1], x_state[:, 2], c='b', marker='o', label='Initial States')

    ax.set_title("Inference of Goal States (3D Scatter Plot)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.legend()

    plt.show()

def cal_error_h(true_states, predict_states, predict_steps):
    true_steps = true_states.shape[0]
    errors_list = []
    stds_list = []
    max_errors_list = []

    for dim in range(true_states.shape[1]):
        errors = []
        for step in range(true_steps - predict_steps):  
            true_values = true_states[step:step + predict_steps, dim]
            predict_values = predict_states[step, :predict_steps, dim]
            error = np.abs(true_values - predict_values)
            errors.append(np.mean(error))
        errors_list.append(np.mean(errors))
        stds_list.append(np.std(errors))
        max_errors_list.append(np.max(errors))
    return errors_list, stds_list, max_errors_list

from shared_autonomy_history import *
def MPC_prediction():
    selected_timestamp = 1
    timestamps = get_available_state_files()
    states, actions = read_states(timestamps[selected_timestamp])
    
    n_state = 10
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

    humanmodel = HumanMPC(goal_weights= [0.86067986, 0.91357833, 1.3121516, 0.63025314, 0.5201712, 0.60911137,
                                          0.7222595, 0.03662273, 0.04287567, -0.19597986],
                          ctrl_weights= [0.93935597, 0.9478975,  0.94227266, 0.95469636], T_HORIZON=100)

    step_idx = 0
    state = states[0]
    state_array=[]
    aim_goal_array=[]
    store_predict_array=[]
    hold_u_x_array=[]

    statess = collections.deque([state[:10]] * 100, maxlen=3)  # Adjust length as needed
    actionss = collections.deque([np.zeros(4)] * 100, maxlen=3)  # Same length as actions array
    for step_idx in range(len(states) - 1):
        print(f"Step {step_idx}")
        # Sample an action in [-1, 1] and scale to environment bounds
        env_act = actions[step_idx]

        # Store the current state and action into the stacks (deques) v 
        statess.append(state[:10])
        actionss.append(env_act)

        aim_goal = humanmodel.run(np.array(statess), np.array(actionss))

        show_env = Quadrotor_v0(0.01)
        show_env.set_state(state[:10])
        hold_u_x = []
        hold_u_x.append(state[:10])
        for _ in range(100):
            hold_u_x.append(show_env.run(env_act))
        hold_u_x = np.array(hold_u_x)
        hold_u_x_array.append(hold_u_x)

        x, u =humanmodel.step(state[:10],aim_goal)
        store_predict_array.append(np.squeeze(x))

        state_array.append(state[:10])
        aim_goal_array.append(aim_goal)

        next_state = states[step_idx + 1]

        state = next_state

    state_array = np.array(state_array)
    aim_goal_array = np.array(aim_goal_array)
    store_predict_array = np.array(store_predict_array)
    hold_u_x_array = np.array(hold_u_x_array)
    np.savez_compressed(
        './Personalized_SA/visualization/threshold_vel_true_human.npz',
        state_array=state_array,
        aim_goal_array=aim_goal_array,
        store_predict_array=store_predict_array,
        hold_u_x_array=hold_u_x_array,
        gate_positions=[]
    )

    data = np.load('./Personalized_SA/visualization/threshold_vel_true_human.npz')
    state_array = data['state_array']
    aim_goal_array = data['aim_goal_array']
    store_predict_array = data['store_predict_array']
    hold_u_x_array = data['hold_u_x_array']
    gate_positions = data['gate_positions']

    plot_state_3d(state_array,
            store_predict_array,
            hold_u_x_array,
            gate_positions)
    
    # Create lists to store error data for both models
    steps_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']
    
    # Dictionary to store metrics for store_predict_array
    store_metrics = {
        'State': state_names
    }
    
    # Dictionary to store metrics for hold_u_x_array
    hold_metrics = {
        'State': state_names
    }
    
    # Calculate errors for each prediction horizon
    for steps in steps_list:
        print(f"{steps} steps:")
        store_error, store_std, store_max = cal_error_h(state_array, store_predict_array, steps)  # 添加最大误差
        hold_error, hold_std, hold_max = cal_error_h(state_array, hold_u_x_array, steps)  # 添加最大误差
        
        # Add to dictionaries
        store_metrics[f'{steps}_steps_mean'] = store_error
        store_metrics[f'{steps}_steps_std'] = store_std
        store_metrics[f'{steps}_steps_max'] = store_max  # 添加最大误差
        hold_metrics[f'{steps}_steps_mean'] = hold_error
        hold_metrics[f'{steps}_steps_std'] = hold_std
        hold_metrics[f'{steps}_steps_max'] = hold_max  # 添加最大误差

    # Save store_predict_array metrics to CSV
    import csv
    store_csv_path = './Personalized_SA/visualization/true_human_store_predict_metrics.csv'
    with open(store_csv_path, 'w', newline='') as csvfile:
        # Create header row
        fieldnames = ['State']
        for steps in steps_list:
            fieldnames.extend([f'{steps}_steps_mean', f'{steps}_steps_std', f'{steps}_steps_max'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each row
        for i, state_name in enumerate(state_names):
            row = {'State': state_name}
            for steps in steps_list:
                row[f'{steps}_steps_mean'] = store_metrics[f'{steps}_steps_mean'][i]
                row[f'{steps}_steps_std'] = store_metrics[f'{steps}_steps_std'][i]
                row[f'{steps}_steps_max'] = store_metrics[f'{steps}_steps_max'][i]  # 添加最大误差
            writer.writerow(row)
    
    # Save hold_u_x_array metrics to CSV
    hold_csv_path = './Personalized_SA/visualization/true_human_hold_u_x_metrics.csv'
    with open(hold_csv_path, 'w', newline='') as csvfile:
        # Create header row
        fieldnames = ['State']
        for steps in steps_list:
            fieldnames.extend([f'{steps}_steps_mean', f'{steps}_steps_std', f'{steps}_steps_max'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each row
        for i, state_name in enumerate(state_names):
            row = {'State': state_name}
            for steps in steps_list:
                row[f'{steps}_steps_mean'] = hold_metrics[f'{steps}_steps_mean'][i]
                row[f'{steps}_steps_std'] = hold_metrics[f'{steps}_steps_std'][i]
            writer.writerow(row)

import seaborn as sns
def visualization():
    file_names = [
        f'./Personalized_SA/visualization/true_human_hold_u_x_metrics.csv',
        f'./Personalized_SA/visualization/true_human_store_predict_metrics.csv',
        f'./Personalized_SA/visualization/true_human.csv'
    ]
    
    # 读取数据
    dataframes = []
    labels = ['Hold U X', 'Store Predict', 'NN']
    
    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)
            dataframes.append(df)
        except FileNotFoundError:
            print(f"文件 {file_name} 未找到")
            return
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # 定义步数和状态
    steps = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 步数列表
    states_to_plot = ['x', 'y', 'z', 'qw', 'qx', 'qy','qz','vx','vy','vz']  # 选择10个状态进行绘图
    
    # 创建子图 - 为legend留出空间
    fig, axes = plt.subplots(1, 10, figsize=(22, 3))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 三种不同颜色
    
    # 用于存储legend的handles和labels
    legend_handles = []
    legend_labels = []
    
    for idx, state in enumerate(states_to_plot):
        ax = axes[idx]
        
        # 为每个数据源绘制线条
        for df_idx, (df, label) in enumerate(zip(dataframes, labels)):
            # 提取该状态的数据
            state_data = df[df['State'] == state]
            
            if len(state_data) == 0:
                continue
                
            means = []
            stds = []
            
            # 提取每个步数的均值和标准差
            for step in steps:
                mean_col = f'{step}_steps_mean'
                std_col = f'{step}_steps_std'
                
                if mean_col in state_data.columns and std_col in state_data.columns:
                    means.append(state_data[mean_col].iloc[0])
                    stds.append(state_data[std_col].iloc[0])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            
            # 绘制均值线和误差带
            line = ax.plot(steps, means, 'o-', label=label, color=colors[df_idx], 
                          linewidth=2, markersize=6)
            # ax.fill_between(steps, means - stds, means + stds, 
            #                alpha=0.2, color=colors[df_idx])
            
            # 只在第一个子图时收集legend信息
            if idx == 0:
                legend_handles.append(line[0])
                legend_labels.append(label)
        
        # 设置子图属性
        ax.set_xlabel('Steps', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.set_title(f'{state}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # 移除每个子图的legend
        # ax.legend(fontsize=10)
        
        # 设置x轴刻度
        ax.set_xticks(steps)
        ax.set_xticklabels(steps, fontsize=8)
        
        # 调整y轴标签字体大小
        ax.tick_params(axis='y', labelsize=8)
    
    # 添加全局legend
    fig.legend(legend_handles, legend_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.95),
               ncol=3, 
               fontsize=12,
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # 调整布局 - 为legend和title留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)  # 为legend留出空间
    
    # 添加主标题
    plt.suptitle(f'Comparison of Different Methods (true_human)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图片
    store_csv_path = './Personalized_SA/visualization/true_human_comparison.png'
    plt.savefig(store_csv_path, dpi=300, bbox_inches='tight')
    plt.show()





# ----------------------------------------------------------------------------------------------------------------------------------------
def calculate_position_velocity_errors(predictions, targets, predict_steps_list):
    """
    计算位置和速度的向量误差（二范数）
    
    Args:
        predictions: 预测值 (num_sequences, predict_steps, state_dim)
        targets: 目标值 (num_sequences, predict_steps, state_dim)
        predict_steps_list: 要评估的预测步数列表
    
    Returns:
        results: 包含位置和速度误差的字典
    """
    results = {}
    
    for steps in predict_steps_list:
        # 提取位置维度 (x, y, z)
        pred_positions = predictions[:, :steps, 0:3]  # x, y, z
        true_positions = targets[:, :steps, 0:3]
        
        # 提取速度维度 (vx, vy, vz)
        pred_velocities = predictions[:, :steps, 7:10]  # vx, vy, vz
        true_velocities = targets[:, :steps, 7:10]
        
        # 计算位置误差（欧几里得距离）
        position_errors = np.linalg.norm(pred_positions - true_positions, axis=2)  # (num_sequences, steps)
        position_mean_error = np.mean(position_errors)
        position_std_error = np.std(position_errors)
        position_max_error = np.max(position_errors)
        
        # 计算速度误差（欧几里得距离）
        velocity_errors = np.linalg.norm(pred_velocities - true_velocities, axis=2)  # (num_sequences, steps)
        velocity_mean_error = np.mean(velocity_errors)
        velocity_std_error = np.std(velocity_errors)
        velocity_max_error = np.max(velocity_errors)
        
        results[steps] = {
            'position': {
                'mean_error': position_mean_error,
                'std_error': position_std_error,
                'max_error': position_max_error
            },
            'velocity': {
                'mean_error': velocity_mean_error,
                'std_error': velocity_std_error,
                'max_error': velocity_max_error
            }
        }
    
    return results

def save_position_velocity_errors_to_csv(results, filename):
    """
    将位置和速度误差保存到CSV文件
    
    Args:
        results: 包含误差数据的字典
        filename: 输出CSV文件名
    """
    import pandas as pd
    
    # 准备数据
    data = []
    predict_steps_list = sorted(results.keys())
    
    # 添加位置误差行
    position_row = {'Metric': 'Position'}
    for steps in predict_steps_list:
        position_row[f'{steps}_steps_mean'] = results[steps]['position']['mean_error']
        position_row[f'{steps}_steps_std'] = results[steps]['position']['std_error']
        position_row[f'{steps}_steps_max'] = results[steps]['position']['max_error']
    data.append(position_row)
    
    # 添加速度误差行
    velocity_row = {'Metric': 'Velocity'}
    for steps in predict_steps_list:
        velocity_row[f'{steps}_steps_mean'] = results[steps]['velocity']['mean_error']
        velocity_row[f'{steps}_steps_std'] = results[steps]['velocity']['std_error']
        velocity_row[f'{steps}_steps_max'] = results[steps]['velocity']['max_error']
    data.append(velocity_row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\nPosition and velocity errors saved to '{filename}'")

def nn_prediction_with_vector_errors():
    """
    运行nn_prediction并额外计算位置和速度的向量误差
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamps = get_available_state_files()
    
    num_train_episodes = 3
    num_episodes = len(timestamps)
    all_episodes_data = []
    
    for selected_timestamp in timestamps:
        states, actions = read_states(selected_timestamp)
        print(f"Loaded states and actions from {selected_timestamp} with shapes: {states.shape}, {actions.shape}")
        all_episodes_data.append((states, actions))
    
    # 数据划分
    train_episodes_idx = list(range(num_train_episodes))
    test_episodes_idx = list(range(num_train_episodes, num_episodes))
    
    print(f"\nUsing episodes {[i+1 for i in train_episodes_idx]} for training")
    print(f"Using episodes {[i+1 for i in test_episodes_idx]} for testing")
    
    # 准备训练数据
    train_episodes_data = [all_episodes_data[i] for i in train_episodes_idx]
    train_sequences, train_targets = combine_multiple_episodes_data(
        train_episodes_data, sequence_length=3, predict_steps=100
    )
    
    print(f"Training data: {len(train_sequences)} sequences from {num_train_episodes} episodes")
    
    # 创建数据加载器
    train_dataset = PredictionDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    input_dim = train_sequences.shape[2]
    model = HumanBehaviorPredictor(input_dim=input_dim, hidden_dim=128, 
                                  num_layers=2, predict_steps=100).to(device)
    
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
                                                       sequence_length=3, predict_steps=100)
        
        for i in range(len(test_sequences)):
            test_data.append((test_sequences[i], test_targets[i]))
    
    print(f"Test data: {len(test_data)} sequences from {len(test_episodes_idx)} episodes")
    
    # 评估模型
    print("\nEvaluating model...")
    predictions, targets = evaluate_model(model, test_data, device)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算位置和速度的向量误差
    predict_steps_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    position_velocity_results = calculate_position_velocity_errors(predictions, targets, predict_steps_list)
    
    # 打印结果
    print("\n" + "="*80)
    print("POSITION AND VELOCITY ERROR SUMMARY")
    print("="*80)
    for steps in predict_steps_list:
        print(f"\n{steps} steps:")
        print(f"  Position - Mean Error: {position_velocity_results[steps]['position']['mean_error']:.6f}, "
              f"Std: {position_velocity_results[steps]['position']['std_error']:.6f}, "
              f"Max: {position_velocity_results[steps]['position']['max_error']:.6f}")
        print(f"  Velocity - Mean Error: {position_velocity_results[steps]['velocity']['mean_error']:.6f}, "
              f"Std: {position_velocity_results[steps]['velocity']['std_error']:.6f}, "
              f"Max: {position_velocity_results[steps]['velocity']['max_error']:.6f}")
    
    # 保存到CSV
    position_velocity_csv_path = 'Personalized_SA/visualization/true_human_position_velocity_errors.csv'
    save_position_velocity_errors_to_csv(position_velocity_results, position_velocity_csv_path)
    
    # # 调用原始的nn_prediction函数以保持兼容性
    # nn_prediction()

def calculate_mpc_position_velocity_errors():
    """
    读取MPC预测数据，计算位置和速度的向量误差并保存到CSV
    """
    # 加载数据
    data = np.load('./Personalized_SA/visualization/threshold_vel_true_human.npz')
    state_array = data['state_array']  # 真实状态
    store_predict_array = data['store_predict_array']  # MPC预测状态
    hold_u_x_array = data['hold_u_x_array']  # 保持控制输入的状态
    
    print(f"Loaded data shapes:")
    print(f"  state_array: {state_array.shape}")
    print(f"  store_predict_array: {store_predict_array.shape}")
    print(f"  hold_u_x_array: {hold_u_x_array.shape}")
    
    # 定义要评估的预测步数
    predict_steps_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # 计算store_predict_array的误差
    store_results = calculate_vector_errors_for_mpc(state_array, store_predict_array, predict_steps_list)
    
    # 计算hold_u_x_array的误差
    hold_results = calculate_vector_errors_for_mpc(state_array, hold_u_x_array, predict_steps_list)
    
    # 保存store_predict_array的结果
    save_mpc_errors_to_csv(store_results, 
                          './Personalized_SA/visualization/true_human_store_predict_position_velocity_errors.csv',
                          'Store Predict')
    
    # 保存hold_u_x_array的结果
    save_mpc_errors_to_csv(hold_results, 
                          './Personalized_SA/visualization/true_human_hold_u_x_position_velocity_errors.csv',
                          'Hold U X')
    
    # 打印结果摘要
    print("\n" + "="*80)
    print("MPC POSITION AND VELOCITY ERROR SUMMARY")
    print("="*80)
    
    print("\nStore Predict Array Results:")
    for steps in predict_steps_list:
        print(f"\n{steps} steps:")
        print(f"  Position - Mean Error: {store_results[steps]['position']['mean_error']:.6f}, "
              f"Std: {store_results[steps]['position']['std_error']:.6f}, "
              f"Max: {store_results[steps]['position']['max_error']:.6f}")
        print(f"  Velocity - Mean Error: {store_results[steps]['velocity']['mean_error']:.6f}, "
              f"Std: {store_results[steps]['velocity']['std_error']:.6f}, "
              f"Max: {store_results[steps]['velocity']['max_error']:.6f}")
    
    print("\n" + "-"*80)
    print("\nHold U X Array Results:")
    for steps in predict_steps_list:
        print(f"\n{steps} steps:")
        print(f"  Position - Mean Error: {hold_results[steps]['position']['mean_error']:.6f}, "
              f"Std: {hold_results[steps]['position']['std_error']:.6f}, "
              f"Max: {hold_results[steps]['position']['max_error']:.6f}")
        print(f"  Velocity - Mean Error: {hold_results[steps]['velocity']['mean_error']:.6f}, "
              f"Std: {hold_results[steps]['velocity']['std_error']:.6f}, "
              f"Max: {hold_results[steps]['velocity']['max_error']:.6f}")

def calculate_vector_errors_for_mpc(true_states, predict_states, predict_steps_list):
    """
    计算MPC预测的位置和速度向量误差
    
    Args:
        true_states: 真实状态 (num_steps, state_dim)
        predict_states: 预测状态 (num_steps, predict_horizon, state_dim)
        predict_steps_list: 要评估的预测步数列表
    
    Returns:
        results: 包含位置和速度误差的字典
    """
    results = {}
    num_steps = true_states.shape[0]
    
    for steps in predict_steps_list:
        position_errors_all = []
        velocity_errors_all = []
        
        # 对每个时间步计算误差
        for t in range(num_steps - steps):
            # 真实的未来轨迹
            true_future_positions = true_states[t:t+steps, 0:3]  # x, y, z
            true_future_velocities = true_states[t:t+steps, 7:10]  # vx, vy, vz
            
            # 预测的未来轨迹
            pred_future_positions = predict_states[t, :steps, 0:3]
            pred_future_velocities = predict_states[t, :steps, 7:10]
            
            # 计算每个预测步的位置误差（欧几里得距离）
            for s in range(steps):
                pos_error = np.linalg.norm(pred_future_positions[s] - true_future_positions[s])
                vel_error = np.linalg.norm(pred_future_velocities[s] - true_future_velocities[s])
                position_errors_all.append(pos_error)
                velocity_errors_all.append(vel_error)
        
        # 转换为numpy数组
        position_errors_all = np.array(position_errors_all)
        velocity_errors_all = np.array(velocity_errors_all)
        
        # 计算统计量
        results[steps] = {
            'position': {
                'mean_error': np.mean(position_errors_all),
                'std_error': np.std(position_errors_all),
                'max_error': np.max(position_errors_all)
            },
            'velocity': {
                'mean_error': np.mean(velocity_errors_all),
                'std_error': np.std(velocity_errors_all),
                'max_error': np.max(velocity_errors_all)
            }
        }
    
    return results

def save_mpc_errors_to_csv(results, filename, method_name):
    """
    将MPC的位置和速度误差保存到CSV文件
    
    Args:
        results: 包含误差数据的字典
        filename: 输出CSV文件名
        method_name: 方法名称（用于标识）
    """
    import pandas as pd
    
    # 准备数据
    data = []
    predict_steps_list = sorted(results.keys())
    
    # 添加位置误差行
    position_row = {'Metric': 'Position'}
    for steps in predict_steps_list:
        position_row[f'{steps}_steps_mean'] = results[steps]['position']['mean_error']
        position_row[f'{steps}_steps_std'] = results[steps]['position']['std_error']
        position_row[f'{steps}_steps_max'] = results[steps]['position']['max_error']
    data.append(position_row)
    
    # 添加速度误差行
    velocity_row = {'Metric': 'Velocity'}
    for steps in predict_steps_list:
        velocity_row[f'{steps}_steps_mean'] = results[steps]['velocity']['mean_error']
        velocity_row[f'{steps}_steps_std'] = results[steps]['velocity']['std_error']
        velocity_row[f'{steps}_steps_max'] = results[steps]['velocity']['max_error']
    data.append(velocity_row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n{method_name} position and velocity errors saved to '{filename}'")

if __name__ == "__main__":
    # nn_prediction()
    # identify_human()
    # MPC_prediction()
    # visualization()

    nn_prediction_with_vector_errors()
    # calculate_mpc_position_velocity_errors()