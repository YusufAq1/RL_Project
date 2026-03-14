import argparse
import random
import shutil
import time
import gymnasium
import coverage_gridworld  # must be imported, even though it's not directly referenced
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


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


maps = [
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
    ]
]


def train(name, resume=False): # Name paramter used to organize runs
    # create training environment. 
    # standard generates random maps
    # we used the pre-defined maps from main.py
    train_env = make_vec_env(
        "just_go",
        n_envs=1,
        env_kwargs={"render_mode": None},
    )

    eval_env = make_vec_env(
        "just_go",
        n_envs=1,
        env_kwargs={"render_mode": None},
    )

    # Initialize the PPO algorithm
    # MlpPolicy is a feedforward neural network
    # train_env is the environment to train on
    # tensorboard_log is the directory to save the training logs
    # verbose=1 means that the training process will be printed to the console
    if resume:
        model = PPO.load(f"./models/{name}/final", env=train_env)
        model.tensorboard_log = f"./logs/{name}"
        print(f"Resuming training from ./models/{name}/final.zip")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            tensorboard_log=f"./logs/{name}",
            verbose=1,
        )

    # Create an evaluation callback
    # This will evaluate the model every 10,000 steps
    # and save the best model to a file
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}",
        eval_freq=100000, 
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )


    # start the training process. 
    model.learn(total_timesteps=500000, callback=eval_callback)
    # saves the model at the end of training
    model.save(f"./models/{name}/final")
    print(f"Training complete. Model saved to ./models/{name}/final.zip")

    train_env.close()
    eval_env.close()


def train_best(name="best_agent"):
    """Three-phase curriculum training for the competition agent.

    Phase 1 — just_go (500k steps):  learn systematic exploration on open grid.
    Phase 2 — standard, no-enemy maps (400k steps): learn to navigate walls.
    Phase 3 — standard, all maps incl. enemies (600k steps): learn enemy avoidance.

    Weights carry forward across phases. Final best_agent.zip is saved from
    the phase-3 checkpoint that scored highest on random unseen maps.
    """
    no_enemy_maps = maps[:2]   # maps[0]=open, maps[1]=walls only

    # ── Phase 1: open grid, learn basic exploration ───────────────────────────
    print(f"\n{'='*50}\n  Phase 1 — 500,000 timesteps\n{'='*50}\n")
    p1_train = make_vec_env("just_go", n_envs=1, env_kwargs={"render_mode": None})
    p1_eval  = make_vec_env("just_go", n_envs=1, env_kwargs={"render_mode": None})

    model = PPO(
        "MlpPolicy",
        p1_train,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=f"./logs/{name}",
        verbose=1,
    )
    model.learn(total_timesteps=500_000, callback=EvalCallback(
        p1_eval, best_model_save_path=f"./models/{name}/phase1/",
        log_path=f"./logs/{name}/phase1", eval_freq=20_000,
        n_eval_episodes=10, deterministic=True, verbose=1,
    ), reset_num_timesteps=False)
    model.save(f"./models/{name}/phase1/final")
    model = PPO.load(f"./models/{name}/phase1/best_model", env=p1_train)
    p1_train.close(); p1_eval.close()

    # ── Phase 2: walls, no enemies ────────────────────────────────────────────
    print(f"\n{'='*50}\n  Phase 2 — 400,000 timesteps\n{'='*50}\n")
    p2_train = make_vec_env(
        "standard", n_envs=1,
        env_kwargs={"render_mode": None, "predefined_map_list": no_enemy_maps},
    )
    p2_eval = make_vec_env(
        "standard", n_envs=1,
        env_kwargs={"render_mode": None, "predefined_map_list": no_enemy_maps},
    )
    model.set_env(p2_train)
    model.learn(total_timesteps=400_000, callback=EvalCallback(
        p2_eval, best_model_save_path=f"./models/{name}/phase2/",
        log_path=f"./logs/{name}/phase2", eval_freq=20_000,
        n_eval_episodes=10, deterministic=True, verbose=1,
    ), reset_num_timesteps=False)
    model.save(f"./models/{name}/phase2/final")
    model = PPO.load(f"./models/{name}/phase2/best_model", env=p2_train)
    p2_train.close(); p2_eval.close()

    # ── Phase 3: all maps including enemies ───────────────────────────────────
    print(f"\n{'='*50}\n  Phase 3 — 600,000 timesteps\n{'='*50}\n")
    p3_train = make_vec_env(
        "standard", n_envs=1,
        env_kwargs={"render_mode": None, "predefined_map_list": maps},
    )
    p3_eval = make_vec_env(
        "standard", n_envs=1,
        env_kwargs={"render_mode": None, "predefined_map_list": None},
    )
    model.set_env(p3_train)
    model.learn(total_timesteps=600_000, callback=EvalCallback(
        p3_eval, best_model_save_path=f"./models/{name}/phase3/",
        log_path=f"./logs/{name}/phase3", eval_freq=20_000,
        n_eval_episodes=10, deterministic=True, verbose=1,
    ), reset_num_timesteps=False)
    model.save(f"./models/{name}/phase3/final")
    model = PPO.load(f"./models/{name}/phase3/best_model", env=p3_train)
    p3_train.close(); p3_eval.close()

    # ── Save final submission model ───────────────────────────────────────────
    shutil.copy(f"./models/{name}/phase3/best_model.zip", "best_agent.zip")
    print(f"\nDone! Competition model saved to best_agent.zip")


def evaluate(name):
    # Load from best_model checkpoint if it exists, else try the path directly
    try:
        model = PPO.load(f"./models/{name}/best_model")
    except FileNotFoundError:
        model = PPO.load(name)  # allows: python main.py eval --name best_agent

    test_envs = [
        ("just_go",       {"render_mode": "human"}),
        ("safe",          {"render_mode": "human"}),
        ("sneaky_enemies",{"render_mode": "human"}),
    ]

    for env_id, kwargs in test_envs:
        print(f"\n── {env_id} ──")
        env = gymnasium.make(env_id, **kwargs)
        for i in range(3):
            obs, info = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                done = done or truncated
            coverage = info["total_covered_cells"] / info["coverable_cells"] * 100
            outcome = "CAUGHT" if info["game_over"] else ("DONE" if info["cells_remaining"] == 0 else "TIMEOUT")
            print(f"  Episode {i+1}: {outcome} | Coverage: {coverage:.1f}% | Steps: {steps}")
            time.sleep(1)
        env.close()


# Example: python main.py train --name my_experiment
# Example: python main.py eval --name my_experiment
# Example: python main.py train --name my_experiment --resume

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "train_best", "eval"], help="train, train_best, or eval")
    parser.add_argument("--name", default="experiment", help="name for this run (used for saving/loading)")
    parser.add_argument("--resume", action="store_true", help="resume training from the last checkpoint")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.name, resume=args.resume)
    elif args.mode == "train_best":
        train_best(name=args.name if args.name != "experiment" else "best_agent")
    else:
        evaluate(args.name)


