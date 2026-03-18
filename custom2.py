import numpy as np
import gymnasium as gym

"""
Competition-optimized custom.py for CISC 474 Coverage Gridworld Tournament.

Observation Spaces:
  A — Multi-Channel Binary Grid (competition default)
  B — Agent-Centric Feature Vector (compact alternative)

Reward Functions:
  1 — Exploration-Focused (simple baseline)
  2 — Progress-Based (coverage-proportional)
  3 — Competition-Optimized Safety-Aware (default)
"""

# ── Selectors ─────────────────────────────────────────────────────────────────
# Change these to switch between configurations for experiments.
ACTIVE_OBS_SPACE = "A"
ACTIVE_REWARD_FN = 3


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION SPACES
# ══════════════════════════════════════════════════════════════════════════════

# --- Color lookup table (RGB → integer ID) ---
_COLORS = np.array([
    [0,   0,   0  ],   # 0 = unexplored (BLACK)
    [255, 255, 255],   # 1 = explored   (WHITE)
    [101, 67,  33 ],   # 2 = wall       (BROWN)
    [160, 161, 161],   # 3 = agent      (GREY)
    [31,  198, 0  ],   # 4 = enemy      (GREEN)
    [255, 0,   0  ],   # 5 = danger-unexplored (RED)
    [255, 127, 127],   # 6 = danger-explored   (LIGHT_RED)
], dtype=np.uint8)


def _grid_to_ids(grid: np.ndarray) -> np.ndarray:
    """Convert RGB grid (10,10,3) → integer IDs (10,10) with values 0-6."""
    flat = grid.reshape(100, 3)
    result = np.zeros(100, dtype=np.int32)
    for cid, color in enumerate(_COLORS):
        result[np.all(flat == color, axis=1)] = cid
    return result.reshape(10, 10)


# ── Observation Space A — Multi-Channel Binary Grid ───────────────────────────
# Six binary channels (10×10 each) = 600 floats in [0, 1].
# Channels:  unexplored | explored | wall | agent | enemy | danger
# This is far more NN-friendly than raw integer categories because the network
# doesn't have to learn that "2" means something categorically different from "3".

def _obs_a(grid: np.ndarray) -> np.ndarray:
    ids = _grid_to_ids(grid)  # (10, 10)
    channels = np.zeros((6, 10, 10), dtype=np.float32)
    channels[0] = (ids == 0).astype(np.float32)              # unexplored
    channels[1] = np.isin(ids, [1, 3]).astype(np.float32)    # explored (WHITE + GREY = visited)
    channels[2] = (ids == 2).astype(np.float32)              # wall
    channels[3] = (ids == 3).astype(np.float32)              # agent position
    channels[4] = (ids == 4).astype(np.float32)              # enemy
    channels[5] = np.isin(ids, [5, 6]).astype(np.float32)    # danger zone (any FOV)

    # ── Directional hint: offset to nearest unexplored cell ──────────────
    # Gives the agent an explicit compass pointing toward unexplored territory,
    # breaking oscillation loops on maps with many remaining cells to visit.
    agent_cells = np.argwhere(ids == 3)
    unexplored_cells = np.argwhere(ids == 0)
    if len(agent_cells) > 0 and len(unexplored_cells) > 0:
        ar, ac = agent_cells[0]
        dists = np.abs(unexplored_cells[:, 0] - ar) + np.abs(unexplored_cells[:, 1] - ac)
        nearest = unexplored_cells[np.argmin(dists)]
        direction = np.array([(nearest[0] - ar) / 9.0, (nearest[1] - ac) / 9.0],
                             dtype=np.float32)
    else:
        direction = np.zeros(2, dtype=np.float32)

    return np.concatenate([channels.flatten(), direction])  # (602,)


def _obs_space_a() -> gym.spaces.Space:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(602,), dtype=np.float32)


# ── Observation Space B — Agent-Centric Feature Vector ────────────────────────
# Compact 18-float vector with normalized values in [-1, 1].

def _obs_b(grid: np.ndarray) -> np.ndarray:
    ids = _grid_to_ids(grid)

    # Agent position
    agent_cells = np.argwhere(ids == 3)
    agent_row, agent_col = (agent_cells[0] if len(agent_cells) > 0 else (0, 0))

    # Enemy positions (up to 5)
    enemy_cells = np.argwhere(ids == 4)

    # Coverage stats
    explored = np.sum(np.isin(ids, [1, 3, 6]))
    remaining = np.sum(np.isin(ids, [0, 5]))
    coverable = max(explored + remaining, 1)
    coverage_ratio = float(explored) / coverable
    remaining_norm = float(remaining) / coverable

    # Enemy relative positions (up to 5, padded)
    enemy_feats = np.zeros(10, dtype=np.float32)
    for i, (er, ec) in enumerate(enemy_cells[:5]):
        enemy_feats[i * 2]     = (er - agent_row) / 9.0
        enemy_feats[i * 2 + 1] = (ec - agent_col) / 9.0

    # Local danger: are the 4 adjacent cells in enemy FOV?
    danger = np.zeros(4, dtype=np.float32)
    for j, (nr, nc) in enumerate([
        (agent_row - 1, agent_col),
        (agent_row + 1, agent_col),
        (agent_row,     agent_col + 1),
        (agent_row,     agent_col - 1),
    ]):
        if 0 <= nr < 10 and 0 <= nc < 10 and ids[nr, nc] in (5, 6):
            danger[j] = 1.0

    return np.concatenate([
        [agent_row / 9.0, agent_col / 9.0],
        enemy_feats,
        [coverage_ratio, remaining_norm],
        danger,
    ]).astype(np.float32)


def _obs_space_b() -> gym.spaces.Space:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def observation_space(env) -> gym.spaces.Space:
    if ACTIVE_OBS_SPACE == "A":
        return _obs_space_a()
    else:
        return _obs_space_b()


def observation(grid: np.ndarray):
    if ACTIVE_OBS_SPACE == "A":
        return _obs_a(grid)
    else:
        return _obs_b(grid)


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _reward_fn1(info: dict) -> float:
    """Reward Function 1 — Exploration-Focused (simple baseline).

    Clear, sparse signals: big reward for new cells, harsh penalty for death,
    huge bonus for full coverage.  Small step/revisit penalties keep the agent
    moving forward.
    """
    r = -0.1
    if info["game_over"]:
        r -= 100
    elif info["cells_remaining"] == 0:
        r += 200
    elif info["new_cell_covered"]:
        r += 10
    else:
        r -= 0.3
    return r


def _reward_fn2(info: dict) -> float:
    """Reward Function 2 — Progress-Based.

    Reward scales with coverage ratio so the agent always has gradient signal
    toward higher coverage.  Quadratic completion bonus accelerates learning
    near 100 %.  Death penalty proportional to remaining coverage.
    """
    coverage = info["total_covered_cells"] / info["coverable_cells"]
    steps_ratio = info["steps_remaining"] / 500.0

    if info["game_over"]:
        return -100 * (1 - coverage)
    elif info["cells_remaining"] == 0:
        return 200
    else:
        progress   = coverage * 5
        completion = (coverage ** 2) * 5
        time_pen   = (1 - steps_ratio) * 0.5
        return progress + completion - time_pen


def _reward_fn3(info: dict) -> float:
    """Reward Function 3 — Competition-Optimized Safety-Aware.

    Designed to maximize tournament performance (coverage %, speed tiebreaker).

    Key design choices:
    • Progressive new-cell reward: increases as coverage grows, preventing the
      agent from "giving up" on hard maps once easy cells are taken.
    • Harsh death penalty (-250): survival = more future coverage.
    • Large victory bonus + speed bonus: incentivizes fast completion.
    • Proximity danger penalty: teaches proactive enemy avoidance rather than
      only learning from death (which is a rare, terminal signal).
    • Moderate revisit penalty: keeps the agent exploring, not looping.
    """
    if info["game_over"]:
        return -250.0

    if info["cells_remaining"] == 0:
        # Victory!  Speed bonus rewards faster completion.
        return 500.0 + info["steps_remaining"] * 0.3

    coverage = info["total_covered_cells"] / info["coverable_cells"]

    if info["new_cell_covered"]:
        # Progressive: 10 at 0% coverage → 25 near 100%
        r = 10.0 + coverage * 15.0
    else:
        # Revisit / idle penalty — strong enough to break oscillation loops
        r = -3.0

    # ── Proximity danger penalty ──────────────────────────────────────────
    # -20.0 per adjacent FOV cell: exceeds max new-cell reward (+25) when
    # 2+ neighbours are in FOV, making the agent reliably avoid danger zones.
    agent_row = info["agent_pos"] // 10
    agent_col = info["agent_pos"] % 10
    fov_cells = set()
    for enemy in info["enemies"]:
        for cell in enemy.get_fov_cells():
            fov_cells.add(cell)

    adjacent_danger = sum(
        1 for (nr, nc) in [
            (agent_row - 1, agent_col),
            (agent_row + 1, agent_col),
            (agent_row,     agent_col - 1),
            (agent_row,     agent_col + 1),
        ]
        if (nr, nc) in fov_cells
    )
    r -= adjacent_danger * 20.0

    # Per-step penalty to encourage speed and break idle loops
    r -= 0.2

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