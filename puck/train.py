import torch, argparse, copy, pathlib, datetime, time, os

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def training_cfg(seed: int = 0) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.6, # 0.3 = more stable, 1.0 = very aggressive at trying new values
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
        },
        "seed": seed,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--resume", type=pathlib.Path, help="Resume from an existing model")
    parser.add_argument("--curriculum", default=0, type=float, help="Curriculum level to start training")
    parser.add_argument("log_dir", type=pathlib.Path, nargs="?", default=None)
    parser.add_argument("--video-every", type=int, default=25, help="Record a video every N iterations")
    parser.add_argument("--video-len", type=float, default=10.0, help="Video length in seconds")

    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("--num-envs and --iterations must be positive")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Checkpoint does not exist: {args.resume}")
    print(f"Setting up training with {args.num_envs} environments and {args.iterations} iterations on device {args.device}")

    if args.log_dir is None:
        base = PROJECT_ROOT / "runs" / datetime.datetime.now().strftime("%Y%m%d")
        args.log_dir = base
        for n in range(1, 1000):
            if not (args.log_dir.exists() and any(args.log_dir.iterdir())):
                break
            args.log_dir = base.with_name(f"{base.name}-{n}")
    assert isinstance(args.log_dir, pathlib.Path)
    print(f"Logging to {args.log_dir}")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print("importing genesis...", end=" ", flush=True)
    start = time.perf_counter()
    import genesis as gs
    print(f"Importing genesis took {time.perf_counter() - start:.2f} seconds")
    from genesis_forge.wrappers import RslRlWrapper, VideoWrapper
    from puck.env import PuckEnv

    if args.device == "cpu":
        torch.set_default_device("cpu")
        backend = gs.cpu
    else: backend = gs.gpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")
    cfg = training_cfg()

    puckenv = PuckEnv(num_envs=args.num_envs, headless=True)
    env = VideoWrapper(
        puckenv,
        video_length_sec=10,
        out_dir=str(args.log_dir / "videos"),
        iteration_trigger=lambda i: i > 0 and i % 25 == 0,
        steps_per_iteration=cfg["num_steps_per_env"],
    )

    # Build the environment
    env = RslRlWrapper(env)
    env.build()
    env.reset()
    print("Training model...")
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), str(args.log_dir), device=gs.device)
    runner.add_git_repo_to_log(".")
    runner.learn(
        num_learning_iterations=args.iterations, init_at_random_ep_len=False
    )
    env.close()

if __name__ == "__main__":
    main()
