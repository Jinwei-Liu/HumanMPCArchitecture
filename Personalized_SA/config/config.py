import argparse    
from datetime import datetime
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3000)
parser.add_argument("--log_dir", type=str, default="./Personalized_SA/human_model/runs_quad")
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--hid_shape", nargs=3, type=int, default=[128, 128, 128])
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=2560)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--adaptive_alpha", action="store_true", default=True)
parser.add_argument("--no_adaptive_alpha", dest="adaptive_alpha", action="store_false")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--threshold_vel", type=int, default=2)
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
        1: ([0.90092283,  0.92079085,  1.3631366,   0.16617055,  0.6782347,   0.5569017, 1.0798198,  -0.89727014, -0.1762917,   0.94635606],
            [0.91950536, 1.0036246,  1.0717165,  0.83069456]),
        2: ([5.5969411e-01,  6.4742762e-01,  1.3425535e+00, -6.7092597e-02, 5.7851774e-01,  2.7476832e-01,  1.3581213e+00,  9.1315401e-01, 4.5179841e-04, -3.1501690e-01],
            [0.93467546, 0.9207936,  1.0009099,  0.7802]),
        3: ([0.26507884, 0.96189415, 1.3121976, 0.6107403, 0.18111606, 0.3404058, 1.1088537, -0.44270095, 0.24729586, -0.10584562],
            [0.9446432, 0.8682919, 0.8690304, 0.9767915]),
        4: ([4.7160548e-01,  2.1757884e+00,  1.2288764e+00,  8.3170748e-01, 5.9110194e-01,  4.2550918e-01,  4.9684775e-01,  6.6608697e-01, -8.8014250e-04, -1.3107462e-01],
            [0.94496953, 0.90043503, 0.98253006, 1.027343]),
        5: ([0.44215772,  0.9167447,   1.2566036,   0.55440533,  0.51803297,  0.5302971, 0.63869774,  0.746156, 0.0783285,  -0.09809528],
            [0.9453663,  0.97088516, 0.9456402,  0.9703641]),
        6: ([0.48592252,  1.0113888,   1.2600814,   0.6299009,   0.6006764,   0.5896704,  0.4823741,   0.6857946,   0.02387187, -0.10714254],
            [0.9507001,  0.9647219,  0.94843996, 0.9586912])
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