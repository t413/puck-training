import math, torch, typing
import genesis as gs
from genesis_forge import ManagedEnvironment, terrain
import genesis_forge.managers as gf_mgrs
import genesis_forge.mdp as gf_mdp
from genesis.utils import geom as gs_util_geom

CONTROL_DT = 1 / 50
INITIAL_BODY_POSITION = (0.0, 0.0, 0.231) #initial robot height in standing pose
HEIGHT_RANGE = (0.02, 0.32) # (m)
D2R : typing.Final[float] = math.pi / 180
PITCH_RANGE = (-30 * D2R, 30 * D2R)
ROLL_RANGE = (-50 * D2R, 50 * D2R)
STANDING_JOINT_POSITIONS = {
    "flexor_r": 0.0,
    "hip_r": math.pi / 4,
    "hip_l": -math.pi / 4,
    "knee_r": math.pi / 2,
    "knee_l": -math.pi / 2,
    "flexor_l": 0.0,
}

class PuckEnv(ManagedEnvironment):
    def __init__(
        self,
        num_envs: int = 1,
        dt: float = CONTROL_DT,
        max_episode_length_s: int | None = 20,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-1.5, -1.0, 0.8),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=33,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
        )
        self.terrain_layout = terrain.TerrainLayout(
            self.scene,
            subterrain_size=(24, 24),
            horizontal_scale=0.1,
            grid=[
                [ terrain.Flat(), terrain.Wave(num_waves=10, border_width=1.0, amplitude=0.1) ], #flat, wavy
                [ terrain.Wave(num_waves=100, amplitude=0.02, border_width=1.0), terrain.Wave(num_waves=2, amplitude=0.3, border_width=1.0) ], # rough, big wide gradient waves
            ],
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="robot.xml",
                pos=INITIAL_BODY_POSITION,
                quat=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        if self.scene.viewer is not None:
            self.scene.viewer.follow_entity(self.robot)
        self.camera = self.scene.add_camera(
            pos=(-1.5, -1.0, 0.8),
            lookat=(0.0, 0.0, 0.2),
            res=(640, 480),
            fov=20,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    def config(self):
        self.terrain_manager = gf_mgrs.TerrainManager(self, layout=self.terrain_layout)
        self.robot_manager = gf_mgrs.EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "position": {
                    "fn": gf_mdp.reset.randomize_terrain_position(
                        terrain_manager=self.terrain_manager,
                        height_offset=INITIAL_BODY_POSITION[2],
                    ),
                },
            },
        )
        self.leg_actuators = gf_mgrs.ActuatorManager(
            self,
            joint_names=["flexor_.*", "hip_.*", "knee_.*"],
            default_pos=STANDING_JOINT_POSITIONS,
            kp=20.0,
            kv=0.5,
            max_force={
                "flexor_.*": 6.0,
                "hip_.*": 6.0,
                "knee_.*": 12.0,
            },
        )
        self.leg_actions = gf_mgrs.PositionActionManager(
            self,
            scale={
                "knee_.*": 0.5, #knees get largest range to encourage crouching behavior
                "hip_.*": 0.25,
                "flexor_.*": 0.25,
            },
            use_default_offset=True,
            actuator_manager=self.leg_actuators,
        )

        self.wheel_actuators = gf_mgrs.ActuatorManager(
            self,
            joint_names=["wheel_.*"],
            default_pos=0.0,
            max_force=12.0, #cybergear max torque
        )
        self.wheel_actions = gf_mgrs.TorqueActionManager(
            self,
            scale=2.0,  # clipped to max_force
            actuator_manager=self.wheel_actuators,
        )

        self.velocity_command = gf_mgrs.VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-2.0, 2.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.5, 1.5),
            },
            stopped_probability=0.2,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={ "envs_idx": [0], "arrow_offset": 0.2 },
        )

        self.height_command = gf_mgrs.CommandManager(
            self,
            range=HEIGHT_RANGE,
            resample_time_sec=4.0,
        )
        self.attitude_command = gf_mgrs.CommandManager(
            self, range={"pitch": PITCH_RANGE, "roll": ROLL_RANGE},
            resample_time_sec=3.0,
        )

        def square_linear_error(error: torch.Tensor, scale: float) -> torch.Tensor:
            '''exponential reward with additial linear penalty to provide gradient pressure for large errors'''
            return torch.exp(-(error / scale).square()) - 0.1 * (error.abs() / scale)

        def effective_attitude(name: str) -> torch.Tensor:
            height = self.height_command.command[:, 0]
            lo, hi = HEIGHT_RANGE
            h_n = INITIAL_BODY_POSITION[2]
            scale = torch.where(
                height < h_n,
                (height - lo) / (h_n - lo),
                (hi - height) / (hi - h_n),
            ).clamp(0.0, 1.0)
            return self.attitude_command.get_command(name).reshape(-1) * scale

        def pitch_tracking(env) -> torch.Tensor:
            actual = gs_util_geom.quat_to_xyz(self.robot_manager.base_quat, rpy=True)[:, 1] #y rotation
            target = effective_attitude('pitch')
            return square_linear_error(actual - target, 0.2)
        def roll_tracking(env) -> torch.Tensor:
            actual = gs_util_geom.quat_to_xyz(self.robot_manager.base_quat, rpy=True)[:, 0] #x rotation
            target = effective_attitude('roll')
            return square_linear_error(actual - target, 0.2)

        def hip_flexor_alignment(env) -> torch.Tensor:
            joint_positions = self.leg_actuators.get_dofs_position()
            flexor_indices = [ self.leg_actuators.dofs_names.index(name) for name in ("flexor_r", "flexor_l") ]
            max_flexor_angle = joint_positions[:, flexor_indices].abs().amax(dim=1)
            tolerance = math.pi / 15 #12deg
            excess = torch.relu(max_flexor_angle - tolerance)
            # return torch.exp(-4.0 * (excess / tolerance).square())
            return square_linear_error(-4.0 * excess, tolerance)

        self.reward_mgr = gf_mgrs.RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "command_linear_velocity": {
                    "weight": 1.0,
                    "fn": gf_mdp.rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                "command_angular_velocity": {
                    "weight": 0.5,
                    "fn": gf_mdp.rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                "command_height": {
                    "weight": 2.0,
                    "fn": gf_mdp.rewards.base_height_exp(
                        height_command=self.height_command,
                        entity_manager=self.robot_manager,
                        terrain_manager=self.terrain_manager,
                        sensitivity=0.005,
                    ),
                },
                "command_pitch": { "weight": 1.0, "fn": pitch_tracking },
                "command_roll":  { "weight": 1.0, "fn": roll_tracking },
                "hip_flexor_alignment": { "weight": 0.5, "fn": hip_flexor_alignment },
                # Don't bounce up and down. Kept light, so it doesn't slow the
                # robot down when moving to a new commanded height.
                "lin_vel_z": {
                    "weight": -0.5,
                    "fn": gf_mdp.rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                # Don't roll or wobble back and forth too much
                # Encourage minimal angular velocity in the XY plane (roll and pitch)
                "ang_vel_xy": {
                    "weight": -0.05,
                    "fn": gf_mdp.rewards.ang_vel_xy_l2(entity_manager=self.robot_manager),
                },
                # Hold still, instead of creeping, when commanded to stop
                "stopped_wheel_velocity": {
                    "weight": -0.005,
                    "fn": gf_mdp.rewards.dof_velocity_l2(
                        actuator_manager=self.wheel_actuators,
                    ),
                    "when": self.velocity_command.when_stopped(),
                },
                # Reduce large rates of change -- encourage smoother movements/actions
                "leg_action_rate": {
                    "weight": -0.05,
                    "fn": gf_mdp.rewards.action_rate_l2(action_manager=self.leg_actions),
                },
                # Reduce joint jitters and encourage smooth actions
                "leg_joint_jitter": {
                    "weight": -0.05,
                    "fn": gf_mdp.rewards.action_acceleration_l2(
                        action_manager=self.leg_actions
                    ),
                },
                # Penalize episodes that ended in termination
                "terminated": {
                    "weight": -10.0,
                    "fn": gf_mdp.rewards.terminated(),
                },
            },
        )

        # Terminate when anything but wheels touch ground, otherwise it won't learn to balance
        self.ground_contact = gf_mgrs.ContactManager(
            self,
            entity=self.robot,
            link_names=["body", "flexor.*", "hamstring.*", "shin.*"],
            with_entity=self.terrain,
        )

        self.termination_manager = gf_mgrs.TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": gf_mdp.terminations.timeout(),
                    "time_out": True,
                },
                "ground_contact": {
                    "fn": gf_mdp.terminations.contact_force(
                        contact_manager=self.ground_contact,
                        threshold=1.0,
                    ),
                },
                "collapsed": {
                    "fn": gf_mdp.terminations.base_height_below_minimum(
                        # Well below the lowest commanded height
                        minimum_height=0.07,
                        entity_manager=self.robot_manager,
                    ),
                },
                "out_of_bounds": {
                    "time_out": True,
                    "fn": gf_mdp.terminations.out_of_bounds(
                        terrain_manager=self.terrain_manager,
                    ),
                },
            },
        )

        self.observation_manager = gf_mgrs.ObservationManager(
            self,
            name="policy",
            cfg={
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "height_cmd": {"fn": self.height_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                    "scale": 0.25,
                    "clip": (-100.0, 100.0),
                },
                "attitude_cmd": {"fn": lambda env: torch.stack([effective_attitude("pitch"), effective_attitude("roll")], dim=-1)},
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "leg_dof_position": {
                    "fn": lambda env: self.leg_actions.get_dofs_position(),
                },
                "leg_dof_velocity": {
                    "fn": lambda env: self.leg_actions.get_dofs_velocity(),
                    "scale": 0.05,
                    "clip": (-100.0, 100.0),
                },
                "wheel_dof_velocity": {
                    "fn": lambda env: self.wheel_actions.get_dofs_velocity(),
                    "scale": 0.05,
                    "clip": (-100.0, 100.0),
                },
                "actions": {
                    "fn": gf_mdp.observations.current_actions(),
                },
            },
        )

        # Privileged observations for the PPO critic, which the robot cannot measure directly
        gf_mgrs.ObservationManager(
            self,
            name="privileged",
            cfg={
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                    "clip": (-100.0, 100.0),
                },
                "base_height": {
                    "fn": lambda env: self.robot_manager.base_pos[:, 2:3],
                },
            },
        )
