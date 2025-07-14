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
parser.add_argument("--threshold_vel", type=int, choices=[1, 3, 5], default=5)
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
        1: ([0.9644163, 0.9895913, 1.3155642, -0.48001483, -0.0014637, 0.04012404,
             -0.21776858, 0.24885532, -0.14794788, 0.24406084],
            [0.84310704, 1.0195727,  1.0203135,  1.0087358]),
        3: ([1.0433292, 1.039954, 1.1844419, -0.21072768, 0.17669167, 0.22673894,
              0.12929986, 0.15958567, -0.00123615, 0.06187058],
            [0.9413348, 1.0150563, 1.0574551, 0.9798457]),
        5: ([9.1905951e-01, 9.3122399e-01, 1.0902803e+00, 2.7676991e-01,
             -6.6724122e-03, -8.7446002e-05, -2.7459403e-02, 1.8389798e-03,
             9.9912984e-03, -7.8114577e-02],
            [0.94444656, 1.0395273, 0.9815468, 0.97386855]),
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