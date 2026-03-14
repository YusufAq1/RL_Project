# CLAUDE.md — Coverage Gridworld RL Project

## Project Overview

This is a university course project (CISC 474) for a Reinforcement Learning tournament. The goal is to train an RL agent to explore all cells of a 10×10 grid while avoiding detection by enemies. The agent is evaluated on previously unseen maps during a tournament.

**Due date:** April 2 (trained agent + code + report)
**Tournament:** April 6 (4 unseen maps, top teams advance each round)

---

## Repository Structure

```
project-root/
├── main.py                          # Training, evaluation, and testing (modify the bottom of this file)
├── coverage-gridworld/
│   ├── setup.py
│   └── coverage_gridworld/
│       ├── __init__.py              # Registered environments (DO NOT MODIFY)
│       ├── env.py                   # Environment logic (DO NOT MODIFY)
│       └── custom.py                # ** THE ONLY FILE YOU MODIFY IN THIS PACKAGE **
```

## Critical Constraints

- **NEVER modify `env.py` or `__init__.py`** — the team gets disqualified from the tournament if `env.py` is changed.
- **ALL environment customization goes in `custom.py`** — this controls `observation_space()`, `observation()`, and `reward()`.
- **ALL training/evaluation code goes in `main.py`** — replace the existing loop at the bottom of the file with your PPO training and evaluation logic.
- You may create additional utility/plotting files outside the `coverage_gridworld` package if needed.
- Use **Stable Baselines 3** for the RL algorithm.

## main.py Structure

`main.py` already contains:
- `human_player()` — manual keyboard control (keep for reference/debugging)
- `random_player()` — random actions (keep for reference)
- `maps` — a list of 5 predefined maps of increasing difficulty (useful for training)
- **Bottom section** — the existing loop that creates the env and runs `human_player()`. **Replace this section with your training and evaluation code.**

The bottom of `main.py` currently looks like this:
```python
env = gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=None, activate_game_status=True)
num_episodes = 5

for i in range(num_episodes):
    env.reset()
    done = False
    while not done:
        action = human_player()
        obs, reward, done, truncated, info = env.step(action)
    if done:
        time.sleep(2)
env.close()
```

**Replace it with** PPO training logic (model creation, `model.learn()`, `EvalCallback`, `model.save()`) and an evaluation function. Keep `human_player()`, `random_player()`, and `maps` intact above — they are useful utilities and the `maps` list should be used for `predefined_map_list` during training.

Use `argparse` or a simple flag variable at the top of the file to switch between training mode and evaluation mode, e.g.:
```python
MODE = "train"  # change to "eval" to watch the trained agent
```

---

## Environment Details

### Grid
- 10×10 grid stored as a numpy array of shape `(10, 10, 3)` with `uint8` RGB values.
- Color meanings:
  - `(0, 0, 0)` BLACK — unexplored cell
  - `(255, 255, 255)` WHITE — explored cell
  - `(101, 67, 33)` BROWN — wall
  - `(160, 161, 161)` GREY — agent
  - `(31, 198, 0)` GREEN — enemy
  - `(255, 0, 0)` RED — unexplored cell under enemy surveillance
  - `(255, 127, 127)` LIGHT_RED — explored cell under enemy surveillance

### Actions
Discrete action space `{0, 1, 2, 3, 4}`:
- 0: Left, 1: Down, 2: Right, 3: Up, 4: Stay

### Enemies
- Each enemy has a **field of view of 4 cells** in the direction they face.
- Enemies **rotate counter-clockwise** every step.
- FOV is blocked by walls and other enemies.
- If the agent is in an enemy's FOV → **game over**.

### Episode Termination
An episode ends when:
- Agent is spotted by an enemy (game over).
- All coverable cells are explored (victory).
- 500 steps are reached (timeout).

### Agent Start
Always starts at cell `(0, 0)` (top-left), with that cell already explored.

### Info Dictionary (available in reward function)
```python
info = {
    "enemies": list,              # List of Enemy objects, each with:
                                  #   .x, .y (int): grid position
                                  #   .orientation (int): LEFT=0, DOWN=1, RIGHT=2, UP=3
                                  #   .get_fov_cells() -> list of (row, col) tuples
    "agent_pos": int,             # Flattened position (e.g., cell (2,3) = position 23)
    "total_covered_cells": int,   # Cells covered so far
    "cells_remaining": int,       # Cells left to visit
    "coverable_cells": int,       # Total coverable cells in this map
    "steps_remaining": int,       # Steps left (starts at 500)
    "new_cell_covered": bool,     # Whether this step discovered a new cell
    "game_over": bool,            # Whether the agent was spotted
}
```

### Available Environments
- `"just_go"` — very easy, 0 enemies, no walls
- `"safe"` — easy, 0 enemies, many walls
- `"maze"` — medium, 2 enemies
- `"chokepoint"` — hard, 4 enemies
- `"sneaky_enemies"` — very hard, 5 enemies
- `"standard"` — random map generation each episode

### Predefined Map List
You can pass a list of custom maps to train on diverse layouts:
```python
gymnasium.make("standard", render_mode=None, predefined_map_list=maps)
```
Map format: 10×10 list of lists. Values: `3`=agent (always at 0,0), `2`=wall, `4`=enemy, `0`=unexplored.

---

## Deliverables

### 1. RL Algorithm [5 points]
- Use **PPO** from Stable Baselines 3 (`from stable_baselines3 import PPO`).
- PPO works well with discrete action spaces and is the recommended choice.
- Use `MlpPolicy` for vector observations.

### 2. Two Distinct Observation Spaces [6 points]
Implement **at least two** different observation spaces in `custom.py`. Switch between them for experiments.

**Observation Space A — Simplified Grid Encoding:**
- Convert the RGB grid (300 values) to a single-channel integer encoding (100 values).
- Map each RGB color to an integer ID: unexplored=0, explored=1, wall=2, agent=3, enemy=4, danger_unexplored=5, danger_explored=6.
- Use `gym.spaces.MultiDiscrete(np.full(100, 7, dtype=np.int32))`.
- Rationale: Dramatically reduces input dimensionality while preserving all necessary information.

**Observation Space B — Agent-Centric Feature Vector:**
- Compact vector containing:
  - Agent position (row, col) normalized to [0, 1]
  - For each enemy: relative position (dx, dy) and orientation (one-hot or integer) — if max 5 enemies, that's up to 20 values
  - Number of cells remaining (normalized)
  - Coverage ratio (total_covered / coverable)
  - Steps remaining (normalized)
  - Local neighborhood danger (are adjacent cells in enemy FOV?)
- Use `gym.spaces.Box` with appropriate bounds.
- Rationale: Very compact representation focused on decision-relevant information.
- NOTE: This approach requires access to `info` data during observation construction. You can compute enemy-related features from the grid colors (RED/LIGHT_RED cells indicate FOV) and store any needed state as attributes on the environment.

**Important implementation notes:**
- `observation_space()` receives the `env` object — use it to access `env.grid`, `env.enemy_list`, etc.
- `observation()` receives only the `grid` numpy array.
- The observation returned by `observation()` MUST match the shape defined in `observation_space()`.
- `env.grid` is `uint8` — when creating MultiDiscrete, use `np.full(..., dtype=np.int32)` to avoid overflow.

### 3. Three Unique Reward Functions [9 points]
Implement **at least three** different reward functions. Switch between them for experiments.

**Reward Function 1 — Exploration-Focused:**
```
+10 for new_cell_covered
-100 for game_over
+200 for full coverage (cells_remaining == 0)
-0.1 per step (encourage speed)
-0.3 for stepping on already-explored cell
```

**Reward Function 2 — Progress-Based:**
```
Reward proportional to coverage progress: e.g., (total_covered / coverable) * scale
Bonus that increases as coverage approaches 100%
Penalty for game_over scaled by how much was left
Time pressure: penalty that increases as steps_remaining decreases
```

**Reward Function 3 — Safety-Aware:**
```
+10 for new_cell_covered
-100 for game_over
Proximity penalty: negative reward when agent is adjacent to enemy FOV cells
Reward for moving away from danger zones
Bonus for covering cells near enemies (risky but valuable)
```

### 4. Experiments and Plots [5 points]
- Train each combination: 2 observation spaces × 3 reward functions = **6 experiments minimum**.
- For each experiment, generate plots showing:
  - Episode reward over training steps
  - Coverage percentage over training steps
  - Episode length over training steps
- Use TensorBoard logging (`tensorboard_log="./logs/"`) and/or matplotlib.
- Use `EvalCallback` from Stable Baselines 3 to periodically evaluate and log metrics.
- Save plots as images for the report.

### 5. Best Agent Training [5 points]
- After experiments, identify the best observation space + reward function combination.
- Train the final agent with more timesteps (1M+) and tuned hyperparameters.
- Use curriculum training: start on easier maps, progress to harder ones.
- Train on `predefined_map_list` with diverse layouts (use the maps from `main.py` plus custom ones) to ensure generalization to unseen tournament maps.
- Save the final model with `model.save("best_agent")` — this ZIP file is submitted separately.

### 6. Report (PDF) [10 points from project tasks + 80 points report marking = major grade component]

The report follows a **specific template** provided by the course. It must be **6 pages maximum** (not counting the title page and references). All figures and tables must have informative captions.

**Marking breakdown for the report itself:**
- Problem Formulation: /20
- Algorithm: /25
- Results: /25
- Discussion: /10

**Required structure (follow this exactly):**

#### Title Page (does not count toward 6-page limit)
- Project title
- Author name(s): First M. Last (no titles or degrees)
- Institutional affiliation (Queen's University, CISC 474)
- Semester: Winter 2026

#### Section 1: Introduction and Problem Formulation (/20 marks)
Provide a short introduction to the problem. Then formally describe the MDP components:

- **State / Observation Space**: Describe BOTH observation spaces implemented. State the size of each, whether discrete or continuous, and the range of values. For Observation Space A (simplified grid encoding): explain the mapping from RGB to integer IDs, that it is a 100-element discrete vector with values in {0..6}. For Observation Space B (agent-centric feature vector): list the features included, the vector size, and that it uses continuous values (Box space). Explain the rationale behind each design choice.

- **Action Space**: Discrete, 5 actions {0=Left, 1=Down, 2=Right, 3=Up, 4=Stay}. Straightforward.

- **Reward Scheme**: Describe ALL THREE reward functions with their formulas/logic. Use mathematical notation where appropriate (e.g., R = +10 if new cell covered, -100 if game over, ...). Explain the reasoning behind each reward signal and what behavior it is intended to encourage or discourage.

#### Section 2: Methodology and Algorithm Description (/25 marks)
- Describe **PPO** (Proximal Policy Optimization): what it is, how it works at a high level (actor-critic, clipped surrogate objective, advantage estimation), and why it was selected for this problem (discrete actions, stable training, good sample efficiency).
- Describe the **network architecture**: MlpPolicy from Stable Baselines 3 (feedforward neural network with default hidden layers [64, 64]).
- Describe **hyperparameter selection**: learning rate, n_steps, batch_size, n_epochs, gamma, total timesteps. Discuss why these values were chosen and whether any tuning was performed.
- Describe any **training strategies**: curriculum learning, map diversity (predefined_map_list vs random), number of training environments, etc.

#### Section 3: Results (/25 marks)
- Present **plots for each experiment** (2 obs spaces × 3 reward functions = 6 experiments). Each plot should show episode reward, coverage percentage, and/or episode length over training timesteps.
- **All figures must have informative captions** describing what is shown.
- Discuss **why** you obtained the results you did. What patterns emerge? Which combinations performed well or poorly?
- Discuss the effect of **different hyperparameters** — if you changed a hyperparameter, how did it affect results?
- Compare the 6 experiments directly (overlay plots or summary table).

#### Section 4: Discussion (/10 marks)
- **Which approach was the best and why?** (best observation space + reward function combo)
- **Which approach was the worst and why?**
- What **challenges** were encountered during implementation and how were they overcome?
- What **future work** or improvements could be considered?
- **Group member contributions**: list what each member worked on.

#### References (does not count toward 6-page limit)
- Cite Stable Baselines 3, PPO paper (Schulman et al., 2017), Gymnasium, and any other resources used.
- Format: Last Name, F. M. (Year). Article Title. Journal Title, Pages From - To.

**Report generation notes for Claude Code:**
- Generate the report as a PDF.
- Use a clean, academic style. No excessive formatting or color.
- Embed all plots as figures with captions (e.g., "Figure 1: Episode reward over training steps for Observation Space A with Reward Function 1").
- Keep language concise and technical — this is a 6-page academic report, not a blog post.
- Do NOT include the title page content in the 6-page count.

---

## Training Tips

- **Start simple**: verify on `just_go` first, then add complexity.
- **Keep observations small**: the README explicitly recommends simpler observations than the default RGB grid.
- **Use diverse training maps**: the tournament uses unseen maps, so agents trained on a single map will fail. Use `predefined_map_list` with the 5 maps from `main.py` plus custom ones, or use `"standard"` for random generation.
- **Hyperparameter tuning**: try different learning rates (1e-4 to 1e-3), n_steps (1024–4096), batch sizes (32–256).
- **Monitor with TensorBoard**: run `tensorboard --logdir ./logs/` to track training in real time.
- **EvalCallback saves the best model automatically** — always use it.
- **Training duration**: 200k steps for quick tests, 500k–1M for real experiments, 1M+ for the final agent.

## Setup Commands
```bash
pip install stable-baselines3[extra] tensorboard
pip install -e coverage-gridworld
```

## Tournament Scoring
- Agents are ranked by **coverage percentage** (how many cells explored).
- Ties broken by **speed** (fewer steps = better).
- 4 maps, top half advances each round. Winning team gets 5 bonus points.

---

## Implementation Plan

Execute in this order:

1. **Set up training infrastructure**: Install dependencies, replace the bottom of `main.py` with PPO training logic (model creation, EvalCallback, TensorBoard logging, model.save). Keep `human_player()`, `random_player()`, and `maps` intact.
2. **Implement Observation Space A** (simplified grid) and **Reward Function 1** (exploration-focused) in `custom.py`. Verify training works on `just_go`.
3. **Implement Reward Functions 2 and 3**. Run experiments with Observation Space A × all 3 rewards.
4. **Implement Observation Space B** (agent-centric features). Run experiments with Observation Space B × all 3 rewards.
5. **Generate plots** for all 6 experiments.
6. **Train the best agent** on diverse maps with the best combo, more timesteps, and tuned hyperparameters.
7. **Write the report** as a PDF following the template exactly: Title Page → Introduction and Problem Formulation (obs spaces, action space, reward schemes) → Methodology and Algorithm Description (PPO, architecture, hyperparameters, training strategy) → Results (plots with captions, analysis) → Discussion (best/worst approach, challenges, future work, member contributions) → References. Maximum 6 pages excluding title page and references.
8. **Package for submission**: ZIP of code, ZIP of trained agent (from `model.save()`), PDF report.

## File Output Expectations
- `custom.py` — final version with the best observation space and reward function
- `main.py` — contains training and evaluation logic (replaces the original bottom section)
- `plots/` — directory with experiment plots (PNG/PDF)
- `best_agent.zip` — trained model from Stable Baselines `save()`
- `report.pdf` — comprehensive project report
- All experiment scripts/configs used