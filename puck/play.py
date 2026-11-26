"""
Drive the trained robot around the simulator with a gamepad.

Controls:
    Left stick up/down      drive forward/backward
    Left stick left/right   turn
    Right stick up/down     raise/lower the body (it returns to the middle height when released)
    Right stick left/right  pitch the body: right tips the nose down, left tips it up
"""

import argparse
import pickle
import sys
from pathlib import Path

import genesis as gs
import torch
from env import PuckEnv
from rsl_rl.runners import OnPolicyRunner

from genesis_forge.gamepads import Gamepad
from genesis_forge.wrappers import RslRlWrapper, VideoWrapper

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default="puck")
parser.add_argument(
    "-r", "--record", action="store_true", help="Record the whole session to a video"
)
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default=None,
    help="e.g. model_249.pt (default: latest)",
)
args = parser.parse_args()

THIS_DIR = Path(__file__).resolve().parent


def get_latest_model(log_dir: Path) -> str:
    """
    Get the last model from the log directory
    """
    checkpoints = list(log_dir.glob("model_*.pt"))
    if not checkpoints:
        print(f"Error: No model files found at '{log_dir}'.")
        sys.exit(1)
    # Sort by the file with the highest number
    latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    return str(latest)


def main():
    # Connect to gamepad
    print("🎮 Connecting to gamepad...")
    gamepad = Gamepad()
    if not gamepad.wait_for_connection(timeout=10):
        print("Error: No gamepad found. Connect one and try again.")
        sys.exit(1)

    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    log_path = Path("./logs") / args.exp_name
    with open(log_path / "cfgs.pkl", "rb") as f:
        [cfg] = pickle.load(f)
    model = (
        str(log_path / args.checkpoint)
        if args.checkpoint
        else get_latest_model(log_path)
    )

    # Setup environment, without an episode time limit
    base_env = PuckEnv(num_envs=1, headless=False, max_episode_length_s=None, mode="play")
    env = base_env
    if args.record:
        print(f"🎥 Recording video to ${str(THIS_DIR)}...")
        env = VideoWrapper(
            base_env,
            filename="play.mp4",
            out_dir=str(THIS_DIR),
            step_trigger=lambda _: True,
            video_length_sec=None,
        )
    env.build()

    # The left stick drives and turns.
    base_env.velocity_command.use_gamepad(
        gamepad,
        lin_vel_x_axis=1,
        ang_vel_z_axis=0,
        lin_vel_y_axis=None,
    )
    # The right stick up/down controls height
    base_env.height_command.use_gamepad(gamepad, range_axis=3, invert_axis=True)
    # The right stick left/right controls roll
    base_env.attitude_command.use_gamepad(gamepad, roll_axis=2)



    print(
        "\n"
        "  Left stick up/down      drive forward/backward\n"
        "  Left stick left/right   turn\n"
        "  Right stick up/down     raise/lower the body\n"
        "  Right stick left/right  roll the body\n"
    )

    # Load the policy
    print(f"Loading {model}...")
    env = RslRlWrapper(env)
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if str(e) != "Viewer closed.":
            raise
    finally:
        env.close()
        gamepad.stop()


if __name__ == "__main__":
    main()
