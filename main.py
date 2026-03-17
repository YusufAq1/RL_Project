import random
import time
import numpy as np
import gymnasium
import shutil
import argparse
import os
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
    """Logs average coverage percentage from info dicts every N steps."""

    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._coverage_buffer = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "coverable_cells" in info and info["coverable_cells"] > 0:
                cov = info["total_covered_cells"] / info["coverable_cells"]
                self._coverage_buffer.append(cov)
        if self.num_timesteps % self.log_freq == 0 and self._coverage_buffer:
            avg_cov = np.mean(self._coverage_buffer)
            self.logger.record("custom/avg_coverage", avg_cov)
            self._coverage_buffer.clear()
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(name, resume=False):
    """Run a single experiment (for report: vary obs space & reward fn in custom.py).
    
    CPU-optimized: 4 envs, ~30-45 min for 1M steps.
    """

    train_env = make_vec_env(
        "standard",
        n_envs=4,
        env_kwargs={"predefined_map_list": all_maps},
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
        total_timesteps=1_000_000,
        callback=[eval_callback, cov_callback],
    )
    model.save(f"./models/{name}/final")
    print(f"Training complete: {name}")

    train_env.close()
    eval_env.close()


def train_curriculum(name="best_agent2"):
    """Train with 4-stage curriculum: easy → hard maps.

    Stage 1: just_go + safe  (0 enemies, walls only)      → 800k steps
    Stage 2: + maze          (2 enemies)                   → 800k steps
    Stage 3: + chokepoint    (4 enemies)                   → 900k steps
    Stage 4: all 16 maps     (up to 5 enemies)             → 1.5M steps
    Total: ~4M steps, ~2–3 hours on CPU.
    """
    os.makedirs(f"./models/{name}", exist_ok=True)

    # Each tuple: (map subset, timesteps for this stage)
    curriculum = [
        (maps[0:2],  800_000),    # Stage 1: just_go + safe  (0 enemies)
        (maps[0:3],  800_000),    # Stage 2: + maze           (2 enemies)
        (maps[0:4],  900_000),    # Stage 3: + chokepoint     (4 enemies)
        (all_maps,   1_500_000),  # Stage 4: all 16 maps      (5 enemies max)
    ]

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


def train_best(name="best_agent"):
    """Train the competition agent with maximum performance settings.

    CPU-optimized: 4 parallel envs, 2M timesteps (~60-90 min on a modern CPU).
    If you have time, increase total_timesteps to 3_000_000 or even 5_000_000.

    Strategy:
    • 16 diverse maps (5 standard + 11 custom) for generalization
    • 4 parallel environments (CPU-optimal)
    • [256, 256] network for spatial reasoning capacity
    • Learning rate linear decay for stable convergence
    • Higher gamma (0.995) because episodes can be 500 steps
    • Moderate entropy (0.02) to maintain exploration
    • Frequent evaluation (every 25k steps, 20 episodes) to catch the best model
    """

    train_env = make_vec_env(
        "standard",
        n_envs=4,
        env_kwargs={"predefined_map_list": all_maps},
    )

    eval_env = make_vec_env(
        "standard",
        n_envs=1,
        env_kwargs={"predefined_map_list": maps},
    )

    os.makedirs(f"./models/{name}", exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=f"./logs/{name}",
        learning_rate=lambda progress: 3e-4 * (0.3 + 0.7 * progress),  # decays to 30% of initial
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
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

    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, cov_callback],
    )
    model.save(f"./models/{name}/final")
    print(f"Best agent training complete: {name}")

    # Copy best model to project root for easy submission
    best_src = f"./models/{name}/best_model.zip"
    if os.path.exists(best_src):
        shutil.copy(best_src, f"./{name}.zip")
        print(f"Best model copied to ./{name}.zip")

    train_env.close()
    eval_env.close()


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(name, render=True):
    """Evaluate the trained agent on multiple environments."""

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

    test_envs = [
        ("just_go",        {"render_mode": render_mode}),
        ("safe",           {"render_mode": render_mode}),
        ("maze",           {"render_mode": render_mode}),
        ("chokepoint",     {"render_mode": render_mode}),
        ("sneaky_enemies", {"render_mode": render_mode}),
    ]

    results = {}

    for env_id, kwargs in test_envs:
        print(f"\n{'='*50}")
        print(f"  {env_id}")
        print(f"{'='*50}")
        env = gymnasium.make(env_id, **kwargs)
        env_results = []

        for ep in range(1):
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
            env_results.append((outcome, coverage, steps))
            print(f"  Ep {ep+1}: {outcome:8s} | Coverage: {coverage:5.1f}% | Steps: {steps}")
            if render:
                time.sleep(0.5)

        env.close()

        avg_cov = np.mean([r[1] for r in env_results])
        completions = sum(1 for r in env_results if r[0] == "COMPLETE")
        results[env_id] = {"avg_coverage": avg_cov, "completions": completions}
        print(f"  → Avg coverage: {avg_cov:.1f}%  |  Completions: {completions}/5")

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for env_id, r in results.items():
        print(f"  {env_id:20s}  avg={r['avg_coverage']:5.1f}%  completed={r['completions']}/5")


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
    args = parser.parse_args()

    if args.mode == "train":
        train(args.name, resume=args.resume)
    elif args.mode == "curriculum":
        train_curriculum(name=args.name if args.name != "experiment" else "best_agent")
    elif args.mode == "train_best":
        train_best(name=args.name if args.name != "experiment" else "best_agent")
    else:
        evaluate(args.name, render=not args.no_render)