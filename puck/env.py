import math
from dataclasses import dataclass
from typing import Literal

import genesis as gs
from genesis.utils.geom import transform_by_quat

from genesis_forge import ManagedEnvironment, terrain
from genesis_forge.managers import (
    ActuatorManager,
    AttitudeCommandManager,
    CommandManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    TerrainManager,
    TorqueActionManager,
    VelocityCommandManager,
)
from genesis_forge.mdp import observations, reset, rewards, terminations

# Distance from the body origin to the ground, when the legs are in the default pose.
# This is a half crouch, so both legs can move up or down to reach any commanded height.
STANDING_HEIGHT = 0.18

# Drop the robot from a few millimeters up, so the wheels don't start inside the ground
INITIAL_BODY_POSITION = (0.0, 0.0, STANDING_HEIGHT + 0.005)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)

EnvMode = Literal["train", "eval", "play"]


class PuckEnv(ManagedEnvironment):
    """
    Two-wheeled balancing robot, with a wheel at the end of each leg,
    trained to follow forward/backward and turning velocity commands,
    and body height and roll commands.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless: bool = True,
        mode: EnvMode = "train",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )
        self.mode = mode

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-1.5, -1.0, 0.8),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
        )

        # Create terrain
        self.terrain_layout = terrain.TerrainLayout(
            self.scene,
            subterrain_size=(24, 24),
            horizontal_scale=0.1,
            grid=[
                [
                    terrain.Flat(),
                    terrain.Wave(num_waves=10, border_width=1.0),
                ],
            ],
        )

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="../robot.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Update the main viewer to follow the robot
        if self.scene.viewer is not None:
            self.scene.viewer.follow_entity(self.robot)

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(-1.5, -1.0, 0.8),
            lookat=(0.0, 0.0, 0.2),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    def config(self):
        """
        Configure the environment managers
        """

        ##
        # Terrain manager handles the terrain layout and updates
        self.terrain_manager = TerrainManager(self, layout=self.terrain_layout)

        ##
        # Robot manager
        # i.e. what to do with the robot when it is reset
        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "position": {
                    "fn": (
                        reset.position(position=(12.0, 24.0, INITIAL_BODY_POSITION[2]))
                        if self.mode == "play"
                        else reset.randomize_terrain_position(
                            terrain_manager=self.terrain_manager,
                            height_offset=INITIAL_BODY_POSITION[2],
                        )
                    ),
                },
            },
        )

        ##
        # Legs: position controlled around a half crouch, which is roughly the
        # middle of the height range and keeps the weight over the wheels.
        # The flexor and hip are EL05 motors (6 Nm), the knee is a Cybergear (12 Nm).
        self.leg_actuators = ActuatorManager(
            self,
            joint_names=["flexor_.*", "hip_.*", "knee_.*"],
            default_pos={
                "flexor_r": 0.0,
                "flexor_l": 0.0,
                "hip_r": 0.42,
                "hip_l": -0.42,
                "knee_r": 1.22,
                "knee_l": -1.22,
            },
            kp=20.0,
            kv=0.5,
            max_force={
                "flexor_.*": 6.0,
                "hip_.*": 6.0,
                "knee_.*": 12.0,
            },
        )
        self.leg_actions = PositionActionManager(
            self,
            # The flexors get less range, so splaying the legs isn't an
            # easy way to change height.
            scale={
                "knee_.*": 0.5,
                "hip_.*": 0.4,
                "flexor_.*": 0.25,
            },
            use_default_offset=True,
            actuator_manager=self.leg_actuators,
        )

        ##
        # Wheels: torque controlled (Cybergear).
        # Both wheel joints spin around the body's +Y axis, so a positive
        # torque rolls either wheel forward.
        self.wheel_actuators = ActuatorManager(
            self,
            joint_names=["wheel_.*"],
            default_pos=0.0,
            max_force=12.0,
        )
        self.wheel_actions = TorqueActionManager(
            self,
            scale=2.0,  # clipped to max_force
            actuator_manager=self.wheel_actuators,
        )

        ##
        # Commanded direction: forward/backward and turning, no side-to-side.
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-2.0, 2.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.5, 1.5),
            },
            stopped_probability=0.15,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
                "arrow_offset": 0.2,
            },
        )

        ##
        # Commanded body height
        self.height_command = CommandManager(
            self,
            range=(0.12, 0.26),
            resample_time_sec=4.0,
        )

        ##
        # Commanded body roll. Positive roll tips the right side down.
        # Genesis Forge always orders the command columns (pitch, roll),
        # symmetry.ATTITUDE_COMMAND depends on that order.
        self.attitude_command = AttitudeCommandManager(
            self,
            range={
                "pitch": (0.0, 0.0),
                "roll": (-math.radians(45.0), math.radians(45.0)),
            },
            resample_time_sec=4.0,
            level_probability=0.2,
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Reward following the linear velocity command
                "command_linear_velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                        sensitivity=0.25,
                        frame="heading", # adjusts for roll
                    ),
                },
                # Reward following the angular velocity command
                "command_angular_velocity": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                        frame="heading", # adjusts for roll
                    ),
                },
                # Reward following the height command
                "command_height": {
                    "weight": 2.0,
                    "fn": rewards.base_height_exp(
                        height_command=self.height_command,
                        entity_manager=self.robot_manager,
                        terrain_manager=self.terrain_manager,
                        sensitivity=0.005,
                    ),
                },
                # Hold the commanded pitch and roll.
                "command_attitude": {
                    "weight": 1.5,
                    "fn": rewards.command_tracking_attitude(
                        attitude_command=self.attitude_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                # Don't bounce up and down. Kept light, so it doesn't slow the
                # robot down when moving to a new commanded height.
                "lin_vel_z": {
                    "weight": -0.5,
                    "fn": rewards.lin_vel_z_l2(
                        entity_manager=self.robot_manager,
                        frame="heading",  # adjusts for roll
                    ),
                },
                # Don't roll or wobble back and forth too much
                # Encourage minimal angular velocity in the XY plane (roll and pitch)
                "ang_vel_xy": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.robot_manager),
                },
                # Hold still, instead of creeping, when commanded to stop
                "stopped_wheel_velocity": {
                    "weight": -0.005,
                    "fn": rewards.dof_velocity_l2(
                        actuator_manager=self.wheel_actuators,
                    ),
                    "when": self.velocity_command.when_stopped(),
                },
                # Reduce large rates of change -- encourage smoother movements/actions
                "leg_action_rate": {
                    "weight": -0.1,
                    "fn": rewards.action_rate_l2(action_manager=self.leg_actions),
                },
                # Reduce joint jitters and encourage smooth actions
                "leg_joint_jitter": {
                    "weight": -0.1,
                    "fn": rewards.action_acceleration_l2(
                        action_manager=self.leg_actions
                    ),
                },
                # Smooth the wheel torques, so the robot doesn't overcorrect and
                # rock back and forth to hold its speed
                "wheel_action_rate": {
                    "weight": -0.01,
                    "fn": rewards.action_rate_l2(action_manager=self.wheel_actions),
                },
                # Penalize episodes that ended in termination
                "terminated": {
                    "weight": -10.0,
                    "fn": rewards.terminated(),
                },
            },
        )

        ##
        # Anything other than the wheels touching the ground.
        # Without this, the robot learned to kneel on its knees and drive like a
        # tripod, which never has to balance.
        self.ground_contact = ContactManager(
            self,
            entity=self.robot,
            link_names=["body", "flexor.*", "hamstring.*", "shin.*"],
            with_entity=self.terrain,
        )

        ##
        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout(),
                    "time_out": True,
                },
                "ground_contact": {
                    "fn": terminations.contact_force(
                        contact_manager=self.ground_contact,
                        threshold=1.0,
                    ),
                },
                "fall_over": {
                    "fn": terminations.bad_orientation(
                        limit_angle=45.0,
                        entity_manager=self.robot_manager,
                    ),
                },
                "collapsed": {
                    "fn": terminations.base_height_below_minimum(
                        # Well below the lowest commanded height
                        minimum_height=0.07,
                        entity_manager=self.robot_manager,
                    ),
                },
                "out_of_bounds": {
                    "time_out": True,
                    "fn": terminations.out_of_bounds(
                        terrain_manager=self.terrain_manager,
                    ),
                },
            },
        )

        ##
        # Observations
        # What the robot can sense: IMU, motor encoders, and its own commands.
        ObservationManager(
            self,
            name="policy",
            cfg={
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "height_cmd": {"fn": self.height_command.observation},
                "attitude_cmd": {"fn": self.attitude_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                    "scale": 0.25,
                    "clip": (-100.0, 100.0),
                },
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
                    "fn": observations.current_actions(),
                },
            },
        )

        # Privileged observations for the PPO critic, which the robot cannot measure directly
        ObservationManager(
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
