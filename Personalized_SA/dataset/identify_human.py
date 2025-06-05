import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Personalized_SA.env.quadrotor_env import QuadrotorRaceEnv
from Personalized_SA.human_model.sac import SAC_countinuous
from Personalized_SA.dataset.identify_human import visualize_path_and_gates
from Personalized_SA.human_model.rlhuman import scale_to_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default="./Personalized_SA/human_model/runs_quad")
    parser.add_argument("--max_test_steps", type=int, default=2000)
    parser.add_argument("--save_path", type=str, default="./Personalized_SA/human_model/checkpoints/actor.pth")
    parser.add_argument("--load_model", type=str, default="./Personalized_SA/human_model/checkpoints/actor.pth")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = args.log_dir.rstrip("/") + "/" + timestamp

    # train(args)
    test(args, temperature=1.0)

