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
parser.add_argument("--threshold_vel", type=int, choices=[10], default=10)
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
        10: ([ 0.6533801,   1.276967,   1.363931,   0.52555615,  0.18058836,  0.284122,
            0.14372152, -0.05602479,  0.20899041, -0.15273543],
            [0.94932413, 0.91885865, 1.0129749,  0.9347645])
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