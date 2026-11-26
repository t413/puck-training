import argparse, pathlib, time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--randomstart", action="store_true", help="Random start position")
    parser.add_argument("--terminations", action="store_true", help="Keep termination policies")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    print("importing torch...", end=" ", flush=True)
    start = time.perf_counter()
    import torch
    print(f"{time.perf_counter() - start:.2f}s")
    print("importing genesis...", end=" ", flush=True)
    start = time.perf_counter()
    import genesis as gs
    print(f"{time.perf_counter() - start:.2f}s")
    from genesis_forge.wrappers import RslRlWrapper
    from rsl_rl.runners import OnPolicyRunner
    from puck.env import PuckEnv, HEIGHT_RANGE, PITCH_RANGE, ROLL_RANGE
    from puck.train import training_cfg

    if args.device == "cpu":
        torch.set_default_device("cpu")
        backend = gs.cpu
    else: backend = gs.gpu

    gs.init(backend=backend, logging_level="warning")

    base_env = PuckEnv(num_envs=1, headless=False, max_episode_length_s=None)
    env = RslRlWrapper(base_env)
    controller = None

    try:
        from genesis_forge.gamepads import Gamepad
        print("Connecting gamepad...")
        controller = Gamepad()
        if not controller.wait_for_connection(timeout=10): #wait for connection before loading the sim
            parser.error("Error: No gamepad found. Connect one and try again.")

        env.build()

        base_env.velocity_command.use_gamepad(
            controller,
            lin_vel_x_axis=1, #l pitch
            lin_vel_y_axis=5, #throwaway
            ang_vel_z_axis=0, #l roll
        )
        # map axes 4 / 5, the rocker buttons, to ramp the height command up and down
        lo, hi = HEIGHT_RANGE
        height = torch.zeros_like(base_env.height_command.command)
        height.fill_((lo + hi) / 2)
        def get_height(_ids) -> torch.Tensor:
            downv = controller.axis(4) / 2
            upv = controller.axis(5) / 2
            if abs((delta := upv - downv)) >= 0.1:
                height.add_(delta * 0.01)
            return height.clamp_(lo, hi)
        base_env.height_command.use_external_controller(get_height)

        base_env.pitch_command.use_gamepad(controller, range_axis=3, invert_axis=True)
        base_env.roll_command.use_gamepad(controller, range_axis=2)

        # Load trained policy
        print(f"Loading {args.checkpoint}...")
        runner = OnPolicyRunner(
            env,
            training_cfg(),
            str(args.checkpoint.parent),
            device=gs.device,
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=gs.device)

        obs, _ = env.reset()

        for cmd in [base_env.velocity_command, base_env.height_command, base_env.roll_command, base_env.pitch_command]:
            cmd.resample_time_sec = None
        base_env.termination_manager.enabled = args.terminations

        if not args.randomstart: # center robot instead
            grid_width = len(base_env.terrain_layout.grid[0])
            grid_height = len(base_env.terrain_layout.grid)
            subterrain_width, subterrain_height = base_env.terrain_layout.subterrain_size
            map_width = grid_width * subterrain_width
            map_height = grid_height * subterrain_height
            center = torch.tensor((map_width / 2.0, map_height / 2.0, 0.5), dtype=torch.float32, device=gs.device).repeat(1, 1)
            reset_ids = torch.tensor([0], dtype=torch.long, device=gs.device)
            base_env.robot.set_pos(center, envs_idx=reset_ids)

        print("Puck is live! Ctrl-C to quit.")

        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)

    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if str(e) != "Viewer closed.":
            raise
    finally:
        if controller is not None:
            controller.stop()
        env.close()


if __name__ == "__main__":
    main()
