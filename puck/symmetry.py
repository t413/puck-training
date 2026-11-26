"""
Left/right mirroring of the robot's observations and actions, for rsl_rl's symmetry_cfg.

The layouts below must match the order of the terms in env.py's observation managers.
Update them whenever an observation, action, or joint is added, removed, or reordered.
"""

import torch
from tensordict import TensorDict

# Each layout lists, for every mirrored value, the (index, sign) of the original it comes from.

# Left/right (Y) flips on vectors, while rotations flip around the other two axes (X, Z)
VECTOR = [(0, 1), (1, -1), (2, 1)]
ANGULAR = [(0, -1), (1, 1), (2, -1)]

VELOCITY_COMMAND = [(0, 1), (1, -1), (2, -1)]  # lin_vel_x, lin_vel_y, ang_vel_z
SCALAR = [(0, 1)]
ATTITUDE_COMMAND = [(0, 1), (1, -1)]  # (pitch, roll): pitch (around Y) is kept, roll (around X) flips

# Swap legs. Order: flexor, hip, knee (right), then flexor, hip, knee (left)
LEGS = [(3, -1), (4, -1), (5, -1), (0, -1), (1, -1), (2, -1)]
# Swap wheels, which both spin around the body's +Y axis
WHEELS = [(1, 1), (0, 1)]


def _concat(*terms: list[tuple[int, int]]) -> list[tuple[int, int]]:
    layout = []
    for term in terms:
        start = len(layout)
        layout += [(start + i, sign) for i, sign in term]
    return layout


ACTIONS = _concat(LEGS, WHEELS)

POLICY = _concat(
    VELOCITY_COMMAND,
    SCALAR,  # height command
    ATTITUDE_COMMAND,
    ANGULAR,  # angular velocity
    VECTOR,  # projected gravity
    LEGS,  # position
    LEGS,  # velocity
    WHEELS,  # velocity
    ACTIONS,
)

PRIVILEGED = _concat(
    VECTOR,  # linear velocity
    SCALAR,  # base height
)

OBSERVATIONS = {
    "policy": POLICY,
    "privileged": PRIVILEGED,
    "critic": POLICY,
}


def _mirror(values: torch.Tensor, layout: list[tuple[int, int]]) -> torch.Tensor:
    if values.shape[-1] != len(layout):
        raise ValueError(
            f"Cannot mirror {values.shape[-1]} values with a layout of {len(layout)}"
            ". Does symmetry.py match the observations and actions in env.py?"
        )
    index = torch.tensor([i for i, _ in layout], device=values.device)
    sign = torch.tensor(
        [s for _, s in layout], device=values.device, dtype=values.dtype
    )
    return values[..., index] * sign


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    return _mirror(actions, ACTIONS)


def mirror_observations(obs: TensorDict) -> TensorDict:
    mirrored = obs.clone()
    for name in obs.keys():
        mirrored[name] = _mirror(obs[name], OBSERVATIONS[name])
    return mirrored


def mirror_obs_actions(
    env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """
    rsl_rl's data augmentation function: returns the originals, followed by their mirror images.
    """
    if obs is not None:
        obs = torch.cat([obs, mirror_observations(obs)], dim=0)
    if actions is not None:
        actions = torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs, actions
