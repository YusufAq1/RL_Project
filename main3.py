import random
import time
import numpy as np
import gymnasium
import shutil
import argparse
import os
import matplotlib.pyplot as plt
import coverage_gridworld  # must be imported, even though it's not directly referenced
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


def human_player():
    # Write the letter for the desired movement in the terminal/console and then press Enter

    input_action = input()
    if input_action.lower() == "w":
        return 3
    elif input_action.lower() == "a":
        return 0
    elif input_action.lower() == "s":
        return 1
    elif input_action.lower() == "d":
        return 2
    elif input_action.isdigit():
        return int(input_action)
    else:
        return 4


def random_player():
    return random.randint(0, 4)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING MAPS — diverse layouts for tournament generalization
# ══════════════════════════════════════════════════════════════════════════════

maps = [
    # Map 0 — just_go (empty, very easy)
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 1 — safe (walls only, easy)
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    ],
    # Map 2 — maze (corridors, 2 enemies, medium)
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],
    # Map 3 — chokepoint (vertical walls, 4 enemies, hard)
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
    # Map 4 — sneaky_enemies (checkered walls, 5 enemies, very hard)
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
    ],
]

# ── Additional training maps for diversity ────────────────────────────────────

extra_maps = [
    # Map 5 — open field, 1 central enemy
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 6 — two rooms connected by corridor, 2 enemies
    [
        [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 2, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    ],
    # Map 7 — L-shaped walls, 3 enemies
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0, 4, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 8 — scattered walls, 3 enemies spread out
    [
        [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    ],
    # Map 9 — 4 corner enemies, few walls
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 10 — grid pattern walls, 2 enemies (mini sneaky)
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 11 — horizontal corridors, 3 enemies
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 12 — dense walls, 1 enemy (navigation challenge)
    [
        [3, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 2, 0, 2, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 2, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 13 — asymmetric, 3 enemies at edges
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0]
    ],
    # Map 14 — ring of walls, 2 inner enemies
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 4, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 4, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 2, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 15 — 5 enemies, open (hardest training map)
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0]
    ],
]

# Combine all maps for training
all_maps = maps + extra_maps


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CALLBACK — logs coverage % to TensorBoard
# ══════════════════════════════════════════════════════════════════════════════

class CoverageLogCallback(BaseCallback):
    """Logs average coverage percentage from info dicts every N steps.
    Also stores (timestep, coverage) pairs in self.coverage_data for plotting.
    """

    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._coverage_buffer = []
        self.coverage_data = []  # list of (timestep, avg_coverage)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "coverable_cells" in info and info["coverable_cells"] > 0:
                cov = info["total_covered_cells"] / info["coverable_cells"]
                self._coverage_buffer.append(cov)
        if self.num_timesteps % self.log_freq == 0 and self._coverage_buffer:
            avg_cov = np.mean(self._coverage_buffer)
            self.logger.record("custom/avg_coverage", avg_cov)
            self.coverage_data.append((self.num_timesteps, avg_cov))
            self._coverage_buffer.clear()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(name, cov_callback):
    """Generate and save 3 training plots for a completed experiment.

    Reads eval reward and episode length from EvalCallback's evaluations.npz.
    Reads coverage % from CoverageLogCallback's in-memory data.
    Saves a single PNG with 3 subplots to ./plots/{name}.png.
    """
    os.makedirs("./plots", exist_ok=True)

    eval_path = f"./logs/{name}/evaluations.npz"
    if not os.path.exists(eval_path):
        print(f"  [plots] No evaluations.npz found at {eval_path}, skipping.")
        return

    data = np.load(eval_path)
    timesteps   = data["timesteps"]
    mean_reward = data["results"].mean(axis=1)
    mean_length = data["ep_lengths"].mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Experiment: {name}", fontsize=14, fontweight="bold")

    # — Episode reward —
    axes[0].plot(timesteps, mean_reward, color="steelblue")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("Episode Reward over Training Steps")
    axes[0].grid(True, alpha=0.3)

    # — Coverage % —
    if cov_callback.coverage_data:
        cov_ts, cov_vals = zip(*cov_callback.coverage_data)
        axes[1].plot(np.array(cov_ts), np.array(cov_vals) * 100, color="seagreen")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("Coverage Percentage over Training Steps")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    # — Episode length —
    axes[2].plot(timesteps, mean_length, color="tomato")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Mean Episode Length (steps)")
    axes[2].set_title("Episode Length over Training Steps")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"./plots/{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(name, resume=False):
    """Run a single experiment (for report: vary obs space & reward fn in custom.py).

    Trains on the 5 canonical maps for 500k steps — enough to show meaningful
    differences between obs spaces and reward functions without taking too long.
    Change ACTIVE_OBS_SPACE and ACTIVE_REWARD_FN in custom.py between runs.
    """

    train_env = make_vec_env(
        "standard",
        n_envs=4,
        env_kwargs={"predefined_map_list": maps},  # 5 canonical maps for fair comparison
    )

    eval_env = make_vec_env(
        "standard",
        n_envs=1,
        env_kwargs={"predefined_map_list": maps},  # eval on the 5 canonical maps
    )

    if resume:
        model = PPO.load(f"./models/{name}/final", env=train_env)
        model.tensorboard_log = f"./logs/{name}"
        print(f"Resuming training: {name}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=f"./logs/{name}",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.02,
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}/",
        eval_freq=50_000,
        n_eval_episodes=15,
        deterministic=True,
        verbose=1,
    )

    cov_callback = CoverageLogCallback()

    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, cov_callback],
    )
    model.save(f"./models/{name}/final")
    print(f"Training complete: {name}")

    generate_plots(name, cov_callback)

    train_env.close()
    eval_env.close()


def train_curriculum(name="best_agent2", resume=False, extra_timesteps=2_000_000):
    """Train with 4-stage curriculum: easy → hard maps.

    Stage 1: just_go + safe  (0 enemies, walls only)      → 800k steps
    Stage 2: + maze          (2 enemies)                   → 800k steps
    Stage 3: + chokepoint    (4 enemies)                   → 1.5M steps
    Stage 4: all 16 maps     (up to 5 enemies)             → 3.0M steps
    Total: ~6.1M steps, ~4–5 hours on CPU.

    resume=True: skips the curriculum stages and continues training the saved
    final model on all_maps. Weights AND optimizer state are restored, so
    training continues exactly where it left off. TensorBoard curve stays
    continuous. Use extra_timesteps to control how long the resumed run lasts.
    """
    os.makedirs(f"./models/{name}", exist_ok=True)

    eval_env = make_vec_env(
        "standard",
        n_envs=1,
        env_kwargs={"predefined_map_list": maps},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}/",
        eval_freq=25_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    cov_callback = CoverageLogCallback()

    if resume:
        # ── Resume: restore full model (weights + optimizer) and keep training ──
        checkpoint = f"./models/{name}/final"
        print(f"\nResuming from {checkpoint}.zip — {extra_timesteps:,} more steps on all maps")
        train_env = make_vec_env(
            "standard",
            n_envs=4,
            env_kwargs={"predefined_map_list": all_maps},
        )
        model = PPO.load(checkpoint, env=train_env)
        model.tensorboard_log = f"./logs/{name}"
        model.learn(
            total_timesteps=extra_timesteps,
            callback=[eval_callback, cov_callback],
            reset_num_timesteps=False,  # keeps timestep counter continuous
        )
        model.save(f"./models/{name}/final")
        print(f"Resumed training complete: {name}")
        train_env.close()

    else:
        # ── Fresh curriculum run ───────────────────────────────────────────────
        curriculum = [
            (maps[0:2],  800_000),    # Stage 1: just_go + safe  (0 enemies)
            (maps[0:3],  800_000),    # Stage 2: + maze           (2 enemies)
            (maps[0:4],  1_500_000),  # Stage 3: + chokepoint     (4 enemies)
            (all_maps,   3_000_000),  # Stage 4: all 16 maps      (5 enemies max)
        ]

        model = None

        for stage_idx, (stage_maps, stage_steps) in enumerate(curriculum):
            print(f"\n{'='*60}")
            print(f"  Stage {stage_idx + 1}/{len(curriculum)}: "
                  f"{stage_steps:,} steps | {len(stage_maps)} maps")
            print(f"{'='*60}")

            train_env = make_vec_env(
                "standard",
                n_envs=4,
                env_kwargs={"predefined_map_list": stage_maps},
            )

            if model is None:
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=1,
                    tensorboard_log=f"./logs/{name}",
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.995,
                    gae_lambda=0.95,
                    ent_coef=0.02,
                    clip_range=0.2,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                )
            else:
                model.set_env(train_env)

            model.learn(
                total_timesteps=stage_steps,
                callback=[eval_callback, cov_callback],
                reset_num_timesteps=(stage_idx == 0),
            )

            model.save(f"./models/{name}/stage_{stage_idx + 1}")
            print(f"  Stage {stage_idx + 1} checkpoint saved.")
            train_env.close()

        model.save(f"./models/{name}/final")
        print(f"\nCurriculum training complete: {name}")

    best_src = f"./models/{name}/best_model.zip"
    if os.path.exists(best_src):
        shutil.copy(best_src, f"./{name}.zip")
        print(f"Best model copied to ./{name}.zip")

    eval_env.close()

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_episode(model, env, render):
    """Run one episode, return (outcome, coverage%, steps)."""
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        done = done or truncated
    coverage = info["total_covered_cells"] / info["coverable_cells"] * 100
    outcome = (
        "CAUGHT" if info["game_over"]
        else ("COMPLETE" if info["cells_remaining"] == 0 else "TIMEOUT")
    )
    if render:
        time.sleep(0.5)
    return outcome, coverage, steps


def evaluate(name, render=True, include_extra_maps=False):
    """Evaluate the trained agent on the 5 canonical environments.

    include_extra_maps=True: also runs one episode on each of the 11 extra_maps
    (maps 5-15) by passing each individually to the 'standard' env.
    """

    try:
        model = PPO.load(f"./models/{name}/best_model")
        print(f"Loaded best model from ./models/{name}/best_model")
    except FileNotFoundError:
        try:
            model = PPO.load(name)
            print(f"Loaded model from {name}")
        except FileNotFoundError:
            print(f"Could not find model '{name}'. Check the path.")
            return

    render_mode = "human" if render else None
    results = {}

    # ── 5 canonical named environments ────────────────────────────────────────
    canonical = [
        ("just_go",        {"render_mode": render_mode}),
        ("safe",           {"render_mode": render_mode}),
        ("maze",           {"render_mode": render_mode}),
        ("chokepoint",     {"render_mode": render_mode}),
        ("sneaky_enemies", {"render_mode": render_mode}),
    ]

    print(f"\n{'='*50}")
    print("  CANONICAL MAPS")
    print(f"{'='*50}")
    for env_id, kwargs in canonical:
        env = gymnasium.make(env_id, **kwargs)
        outcome, coverage, steps = _run_episode(model, env, render)
        env.close()
        results[env_id] = {"coverage": coverage, "outcome": outcome}
        print(f"  {env_id:20s}  {outcome:8s}  Coverage: {coverage:5.1f}%  Steps: {steps}")

    # ── Extra maps (optional) ──────────────────────────────────────────────────
    if include_extra_maps:
        print(f"\n{'='*50}")
        print("  EXTRA MAPS (5–15)")
        print(f"{'='*50}")
        for i, m in enumerate(extra_maps):
            label = f"extra_map_{i + 5}"
            env = gymnasium.make("standard", render_mode=render_mode,
                                 predefined_map_list=[m])
            outcome, coverage, steps = _run_episode(model, env, render)
            env.close()
            results[label] = {"coverage": coverage, "outcome": outcome}
            print(f"  {label:20s}  {outcome:8s}  Coverage: {coverage:5.1f}%  Steps: {steps}")

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for env_id, r in results.items():
        print(f"  {env_id:20s}  {r['coverage']:5.1f}%  {r['outcome']}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coverage Gridworld RL Agent")
    parser.add_argument(
        "mode",
        choices=["train", "train_best", "curriculum", "eval"],
        help="train: single experiment | curriculum: staged easy→hard | train_best: competition agent | eval: evaluate",
    )
    parser.add_argument("--name", default="experiment", help="run name (for saving/loading)")
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    parser.add_argument("--no-render", action="store_true", help="disable rendering in eval")
    parser.add_argument("--extra-maps", action="store_true", help="also evaluate on extra_maps 5-15")
    parser.add_argument("--extra-timesteps", type=int, default=2_000_000,
                        help="additional timesteps when resuming curriculum (default: 2M)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.name, resume=args.resume)
    elif args.mode == "curriculum":
        curriculum_name = args.name if args.name != "experiment" else "best_agent2"
        train_curriculum(name=curriculum_name, resume=args.resume,
                         extra_timesteps=args.extra_timesteps)
    elif args.mode == "train_best":
        train_best(name=args.name if args.name != "experiment" else "best_agent")
    else:
        evaluate(args.name, render=not args.no_render, include_extra_maps=args.extra_maps)