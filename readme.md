# Puck RL training system 🏋🏼

### your _Personal Upright Control Kompanion_

Puck is an 8DOF bipedal wheeled robot project. This is my attempt to train an RL policy to stand it up and balance!

See also Puck's [MakerWorld design page](https://t413.com/go/puck-mw?ref=puckrl), [Firmware Flash](https://t413.com/p/projects/puck/?ref=puckrl) web tool, my [Discord](https://t413.com/go/discord?ref=puckrl).

![puck gif horizontal](https://t413.com/p/projects/puck/puck-horiz-small.gif)
![training gif](https://t413.com/p/projects/puck/pucktraining.gif)

## Basics

This repository is a _self-contained_ robot-model + training harness. It's goal is to, from scratch each time, train-up a small neural net to drive Puck. It's goal is to bring _determinism_ to the chaos of a new neural net. Think of this as a school's curriculum: it can start fresh with _any_ new student and bring them up to speed. Along the way the training process outputs video files to show the progress, and detailed charts of the progress let you gain insights.

## Running

- Get `uv`: `brew install uv` or `pip install uv`
- `uv run python -m puck.train` trains a policy
  * This outputs to runs/{date}/ by default.
  * Videos of training are saved to runs/{date}/videos/ every few iterations (configurable)
  * View charts of the training progress (live or after) with `uv run tensorboard --logdir runs/[yyyy-mm-dd]`
  * Running through uv auto-installs dependencies to `./.venv`
- `uv run python -m puck.runsim runs/[yyyy-mm-dd]/model_799.pt` to simulate the result, controllable with a gamepad 🎮! (I've been using a PS4 controller)

## Deploying

- Stay tuned!

## About the robot

Puck is an 8DOF bipedal wheeled robot.

- 8 DOF, degrees of freedom, meaning 8 QDD actuators. Each leg has
   * Hip Flexor: Robstride EL05. This swings the leg out/in, letting the legs swing sidewise and/or the body/head have a roll axis.
   * Hip Actuator: Robstride EL05. This swings the leg forward/backward, also letting the body/head have a pitch axis.
   * Knee Actuator: Xiaomi Cybergear. This bends the leg at the knee.
   * Wheel Actuator: Xiaomi Cybergear. This spins the wheel / foot.
- The body is also the head. I'll use these interchangeably.
   * Big googly eyes
   * Super fancy hatch system
   * E-Stop on top
   * Nice recessed snap-in spot for the main controller: M5 Atom S3R
   * Holds the battery, a 6S drone battery with BMS
- MJCF Model is exported by `onshape-to-robot` then modified. See _"How I made this"_ below!
- Actuation specifics
   * Actuators are all driven in MIT mode, with torque, velocity, kp, kd, and position all being set in every packet. MIT mode works because if kp and kd are 0 then position target has zero effect. Otherwise vel/torq are like FF values. With `kp+kd=0` then vel and/or torque targets can work like standard torque or velocity modes.
   * Actuators are all zeroed with the robot _crouching_, its initial position in the MJCF as well.
   * Position targets are in radians. _Right-hand_ rotation is _positive_.
   * Each actuator replies back with position, velocity, and torque state after every command (and/or when requested). These are intended to be fed into the RL model.
- Simulation model specifics
   * The robot model starts out crouching on the ground in its initial zero-position.
   * The env changes this to a standing position for better training.
- Control desires:
   * Height. Ideally expressed as a value from [0 to 1], 0 is crouching, 1 is maximum leg extension.
   * Head/Body pitch & roll. Two independent values. Both expressed from [-1 to 1]
   * Forward velocity in m/s. Referenced to _head/body position in the world_. When a forward velocity target is step-functioned up the robot will have to tilt forward-- the wheels will drive _backwards_. This is good, and starts making the body translate forwards. Cybergears are [0.140mm diam * pi * 300rpm / 60] = 2.2 m/s max (~5mph)
   * HeadBody yaw angular velocity. Referenced in rad/s
   * Side velocity in m/s. This is the difficult one. It requires the robot to _step_ with its wheels lifting off the ground in sequence.
   * Eventually: actions like jump, hop, and behavior modifiers.

## How I made this

- Onshape model prep
   1. Made a new document that imported the full robot design's sketches & planes only
   1. Made an ultra simplified version of Puck, specifically for training:
      a. All leg parts are symmetric so they can easily be placed twice in the assembly later
      a. All parts have _mass override_ and _center-of-mass override_ set. This is vital.
      a. All parts (except wheels) are intentionally very low-polygon for best sim performance
      a. Trying to keep parts convex-only, no concave portions. Simplifies not having to have the separate (auto-generated) "convexify" mesh
      a. Used transform tool to move the limbs to the robot's zero-position
   1. Added _mate connectors_ at each joint, based on _sketch positions_
      a. Two mates for each joint, one attached to each side's body
   1. Added an additional _mate connector_ to a new center, origin position where the wheels are on the ground and the robot is lying flat.
      a. This lets you align the robot to the origin in the assembly!
      a. Mates don't need to be on or touching a body, they can still be attached to that body!
   1. Create assembly
      a. Add all simplified model's parts, adding the symmetric other side as well
      a. Add _Revolute Mates_ for each joint
      a. **Rename** all revolute mates `dof_*` (like dof_flexor_r)
      a. Add _preliminary_ limits to each revolute mate (except wheels)
         - DO NOT perfect these, the export process messes them up and they need to be verified later anyway.
      a. Test moving the system by clicking and dragging. Temporarily "fix" the body for this.
      a. Add a folder of _Fastened mates_ for model alignment before export
         - Add a mate between your "new center origin" position from before to the assembly's origin position
         - Click the "Reorient secondary axis" until your model is **facing positive-x direction**. That's _facing right_ in onshape land.
         - Add a new fastened mate for each joint, fixing into place in the zero position for your bot
      a. Once zero-aligned, right click on the folder and suppress the whole thing.
- OnShape-to-robot export and prep
   1. `uv pip install onshape-to-robot` to get the program ([documentation](https://onshape-to-robot.readthedocs.io/en/latest/))
   1. run `export ONSHAPE_API=https://cad.onshape.com; export ONSHAPE_ACCESS_KEY=on_xyz; export ONSHAPE_SECRET_KEY=woo;` to get your API key setup
   1. Add a `config.json` file with `echo {"output_format": "mujoco", "url": "https://cad.onshape.com/documents/1234/w/5678/e/9101112"}' | tee config.json` (changing the URL to your assembly's actual URL)
   1. run `./.venv/bin/onshape-to-robot .` to export (or update) your model
   1. Edit your resulting `robot.xml`
      a. IMU definition added based on controller position
      a. Setup actuators to be specific to the el05 + cybergear drives
      a. Add wheel friction, actuator sensors, and timestep
   1. Test, verify, and get real joint limits for your bot using `uv run gs launch ./scene.xml` (or just ./robot.xml) using the "Joint Control" area.
