import argparse    
from datetime import datetime
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3000)
parser.add_argument("--log_dir", type=str, default="./Personalized_SA/human_model/runs_quad")
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--hid_shape", nargs=3, type=int, default=[128, 128, 128])
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=2560)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--adaptive_alpha", action="store_true", default=True)
parser.add_argument("--no_adaptive_alpha", dest="adaptive_alpha", action="store_false")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--threshold_vel", type=int, choices=[1, 3, 5], default=3)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--visualization_save_path", type=str, default=None)
parser.add_argument("--goal_weights", type=float, default=None)
parser.add_argument("--ctrl_weights", type=float, default=None)
args = parser.parse_args()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
args.log_dir = args.log_dir.rstrip("/") + "/" + timestamp
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_weights_and_paths(threshold_vel):
    weight_map = {
        1: ([1.1508648, 1.1990767, 1.1594374, 0.70566994, 0.61176234, 0.2175607,
            -0.6986771, -0.7920079, -0.7285001, -0.47435185],
            [0.9479804, 0.02958369, 0.02927853, 0.03684817]),
        3: ([ 1.1479964,   1.0211719,   0.848215,    0.64405197,  0.20096746, -0.00190247,
            -0.48737872, -0.01006169,  0.01761045, -0.01894934],
            [0.8606804,  0.00966638, 0.01236562, 0.01788813]),
        5: ([ 1.5217704,   1.0734304,   1.2253189,   0.56899023,  0.22566086, -0.33834934,
            0.34679148, -0.01452107,  0.02399969, -0.05830049],
            [0.9528509,  0.02332965, 0.02409738, 0.03276937]),
    }
    
    # 根据threshold_vel返回对应的权重
    goal_weights, ctrl_weights = weight_map.get(threshold_vel)

    # 设置路径中的数字
    save_path = f"./Personalized_SA/human_model/checkpoints/actor_{threshold_vel}.pth"
    load_model = f"./Personalized_SA/human_model/checkpoints/actor_{threshold_vel}.pth"
    visualization_save_path = f"./Personalized_SA/visualization/threshold_vel_{threshold_vel}.npz"

    return goal_weights, ctrl_weights, save_path, load_model, visualization_save_path

# 获取根据threshold_vel变化的权重和路径
goal_weights, ctrl_weights, save_path, load_model, visualization_save_path = get_weights_and_paths(args.threshold_vel)

# 更新args中的值
args.goal_weights = goal_weights
args.ctrl_weights = ctrl_weights
args.save_path = save_path
args.load_model = load_model
args.visualization_save_path = visualization_save_path

def set_seed(seed):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # For all GPUs (if multiple GPUs are used)
    
    # Set PyTorch's deterministic behavior for better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42) 