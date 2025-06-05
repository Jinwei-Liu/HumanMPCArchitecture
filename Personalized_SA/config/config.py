import argparse    
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--log_dir", type=str, default="./Personalized_SA/human_model/runs_quad")
parser.add_argument("--max_test_steps", type=int, default=2000)
parser.add_argument("--save_path", type=str, default="./Personalized_SA/human_model/checkpoints/actor.pth")
parser.add_argument("--load_model", type=str, default="./Personalized_SA/human_model/checkpoints/actor.pth")
args = parser.parse_args()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
args.log_dir = args.log_dir.rstrip("/") + "/" + timestamp

