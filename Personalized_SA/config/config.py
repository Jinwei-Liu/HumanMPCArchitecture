import argparse    
from datetime import datetime
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3000)
parser.add_argument("--log_dir", type=str, default="./Personalized_SA/human_model/runs_quad")
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--save_path", type=str, default="./Personalized_SA/human_model/checkpoints/actor_0.pth")
parser.add_argument("--load_model", type=str, default="./Personalized_SA/human_model/checkpoints/actor_0.pth")
parser.add_argument("--hid_shape", nargs=3, type=int, default=[128, 128, 128])
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=2560)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--adaptive_alpha", action="store_true", default=True)
parser.add_argument("--no_adaptive_alpha", dest="adaptive_alpha", action="store_false")
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
args.log_dir = args.log_dir.rstrip("/") + "/" + timestamp
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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