import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


# ── Selectors ─────────────────────────────────────────────────────────────────
# Change these to switch observation space and reward function.
# Obs space: "A" = Simplified Grid Encoding, "B" = Agent-Centric Feature Vector
# Reward fn: 1 = Exploration-Focused, 2 = Progress-Based, 3 = Safety-Aware
ACTIVE_OBS_SPACE = "A"
ACTIVE_REWARD_FN = 1


# ── Observation Space A — Simplified Grid Encoding ────────────────────────────
# Maps each RGB color to an integer ID (0-6), reducing 300 raw values to 100.
_COLORS = np.array([
    [0,   0,   0  ],  # 0 = unexplored (BLACK)
    [255, 255, 255],  # 1 = explored (WHITE)
    [101, 67,  33 ],  # 2 = wall (BROWN)
    [160, 161, 161],  # 3 = agent (GREY)
    [31,  198, 0  ],  # 4 = enemy (GREEN)
    [255, 0,   0  ],  # 5 = unexplored under surveillance (RED)
    [255, 127, 127],  # 6 = explored under surveillance (LIGHT_RED)
], dtype=np.uint8)


def _encode_grid(grid: np.ndarray) -> np.ndarray:
    """Convert RGB grid (10,10,3) to integer ID array of shape (100,)."""
    flat = grid.reshape(100, 3)
    result = np.zeros(100, dtype=np.int32)
    for color_id, color in enumerate(_COLORS):
        result[np.all(flat == color, axis=1)] = color_id
    return result


# ── Observation Space B — Agent-Centric Feature Vector ────────────────────────
# 18-element float32 vector, all values in [-1, 1]:
#   [0:2]   agent position (row/9, col/9)
#   [2:12]  up to 5 enemies: (dr/9, dc/9) relative to agent, padded with 0
#   [12]    coverage ratio (explored / coverable)
#   [13]    cells remaining normalized (remaining / coverable)
#   [14:18] local danger: N/S/E/W neighbor is in enemy FOV (0 or 1)
def _obs_b(grid: np.ndarray) -> np.ndarray:
    encoded = _encode_grid(grid).reshape(10, 10)

    # Agent position
    agent_cells = np.argwhere(encoded == 3)
    agent_row, agent_col = agent_cells[0] if len(agent_cells) > 0 else (0, 0)

    # Enemy positions
    enemy_cells = np.argwhere(encoded == 4)

    # Coverage stats
    explored = np.sum(np.isin(encoded, [1, 3, 6]))   # WHITE, GREY, LIGHT_RED
    remaining = np.sum(np.isin(encoded, [0, 5]))       # BLACK, RED
    coverable = explored + remaining
    coverage_ratio = float(explored) / coverable if coverable > 0 else 0.0
    remaining_norm = float(remaining) / coverable if coverable > 0 else 0.0

    # Enemy relative positions (up to 5, padded with 0)
    enemy_features = np.zeros(10, dtype=np.float32)
    for i, (er, ec) in enumerate(enemy_cells[:5]):
        enemy_features[i * 2]     = (er - agent_row) / 9.0
        enemy_features[i * 2 + 1] = (ec - agent_col) / 9.0

    # Local danger: are the 4 adjacent cells in enemy FOV?
    danger = np.zeros(4, dtype=np.float32)
    for j, (nr, nc) in enumerate([
        (agent_row - 1, agent_col),  # North
        (agent_row + 1, agent_col),  # South
        (agent_row,     agent_col + 1),  # East
        (agent_row,     agent_col - 1),  # West
    ]):
        if 0 <= nr < 10 and 0 <= nc < 10 and encoded[nr, nc] in (5, 6):
            danger[j] = 1.0

    return np.concatenate([
        [agent_row / 9.0, agent_col / 9.0],
        enemy_features,
        [coverage_ratio, remaining_norm],
        danger,
    ]).astype(np.float32)


def observation_space(env: gym.Env) -> gym.spaces.Space:
    if ACTIVE_OBS_SPACE == "A":
        # 100 cells, each with 7 possible integer values (0–6)
        return gym.spaces.MultiDiscrete(np.full(100, 7, dtype=np.int32))
    else:
        # 18-element continuous vector, all values in [-1, 1]
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)


def observation(grid: np.ndarray):
    if ACTIVE_OBS_SPACE == "A":
        return _encode_grid(grid)
    else:
        return _obs_b(grid)


# ── Reward functions ──────────────────────────────────────────────────────────

def _reward_fn1(info: dict) -> float:
    """Reward Function 1 — Exploration-Focused."""
    r = -0.1  # small per-step penalty to encourage speed
    if info["game_over"]:
        r -= 100
    elif info["cells_remaining"] == 0:
        r += 200
    elif info["new_cell_covered"]:
        r += 10
    else:
        r -= 0.3  # light penalty for revisiting, much less than new-cell reward
    return r


def _reward_fn2(info: dict) -> float:
    """Reward Function 2 — Progress-Based.

    Per-step reward scales with current coverage ratio, with a quadratic bonus
    as coverage approaches 100%. Game-over penalty is proportional to how much
    was left uncovered (so dying early hurts more than dying late). A small
    time-pressure penalty grows as the step budget runs out.
    """
    coverage_ratio = info["total_covered_cells"] / info["coverable_cells"]
    steps_ratio = info["steps_remaining"] / 500  # 1.0 at start, 0.0 at timeout

    if info["game_over"]:
        # Dying early (low coverage) is punished more than dying late
        return -100 * (1 - coverage_ratio)
    elif info["cells_remaining"] == 0:
        return 200
    else:
        progress_reward = coverage_ratio * 5          # proportional to coverage
        completion_bonus = (coverage_ratio ** 2) * 5  # accelerates near 100%
        time_pressure = (1 - steps_ratio) * 0.5       # grows as steps run out
        return progress_reward + completion_bonus - time_pressure


def _reward_fn3(info: dict) -> float:
    """Reward Function 3 — Safety-Aware.

    Builds on RF1's exploration signal but adds:
    - A proximity penalty scaling with how many adjacent cells are in enemy FOV.
    - An extra bonus when covering a new cell while adjacent to danger (risky but valuable).
    The proximity penalty alone incentivises moving away from danger zones because
    the penalty disappears when the agent leaves the area.
    """
    agent_row = info["agent_pos"] // 10
    agent_col = info["agent_pos"] % 10

    # Collect all FOV cells from all enemies
    fov_cells = set()
    for enemy in info["enemies"]:
        for cell in enemy.get_fov_cells():
            fov_cells.add(cell)

    # Count how many of the 4 adjacent cells are in enemy FOV
    adjacent_danger = sum(
        1 for (nr, nc) in [
            (agent_row - 1, agent_col),
            (agent_row + 1, agent_col),
            (agent_row,     agent_col - 1),
            (agent_row,     agent_col + 1),
        ]
        if (nr, nc) in fov_cells
    )

    r = -0.1  # base step penalty
    if info["game_over"]:
        r -= 100
    elif info["cells_remaining"] == 0:
        r += 200
    elif info["new_cell_covered"]:
        r += 10
        if adjacent_danger > 0:
            r += 5  # bonus for covering a risky cell near enemy FOV
    else:
        r -= 0.3

    # Proximity penalty: grows with number of dangerous adjacent cells
    r -= adjacent_danger * 1.0

    return r


def reward(info: dict) -> float:
    if ACTIVE_REWARD_FN == 1:
        return _reward_fn1(info)
    elif ACTIVE_REWARD_FN == 2:
        return _reward_fn2(info)
    elif ACTIVE_REWARD_FN == 3:
        return _reward_fn3(info)
    else:
        raise ValueError(f"Unknown ACTIVE_REWARD_FN: {ACTIVE_REWARD_FN}")
