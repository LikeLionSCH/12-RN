import gymnasium as gym
from sb3_contrib import MaskablePPO
import torch

import argparse
import time
import torch
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from MiniChessEnv import MiniChessEnv


def make_env_fn():
    def _init():
        env = gym.make("MiniChess-v0")
        env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
        return env

    return _init


def get_device(force_cpu: bool = False):
    if not force_cpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_vec_env(n_envs: int):
    env_fns = [make_env_fn() for _ in range(n_envs)]
    try:
        env = SubprocVecEnv(env_fns)
    except Exception:
        env = DummyVecEnv(env_fns)
    # Wrap with VecMonitor so episode reward/length statistics are logged
    env = VecMonitor(env)
    return env


def main(quick: bool = False, n_envs: int = 8, force_cpu: bool = False):
    device = get_device(force_cpu)
    print(f"Using device: {device}")

    # small quick mode for smoke test
    up_timesteps = 80_000 if not quick else 2_000
    down_timesteps = 80_000 if not quick else 2_000

    for i in range(1, 100000):
        print("Training model_" + str(i))
        user = 'up' if i % 2 == 0 else 'down'

        enemy = 'down' if user == 'up' else 'up'

        env = create_vec_env(n_envs)

        custom_objects = {
            "clip_range": lambda _: 0.2,
            "lr_schedule": lambda _: 2.5e-4,
        }

        # load enemy model onto same device (if exists)
        try:
            enemy_model = MaskablePPO.load(f"./models/model_{enemy}.zip", env=env, custom_objects=custom_objects, device=device)
            env.envs[0].unwrapped.set_enemy_env(enemy_model, enemy=0 if enemy == 'up' else 1)
        except Exception:
            enemy_model = None

        model_path = f"./models/model_{user}.zip"

        try:
            model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects, device=device)
            model.verbose = 1
            model.ent_coef = 0.01
            model.learning_rate = 2.5e-4
            print(f"Loaded existing model: {model_path} (Continuing training)")
        except Exception:
            print(f"No previous model found at {model_path}, training from scratch")
            policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                batch_size=512,
                learning_rate=2.5e-4,
                ent_coef=0.01,
                n_steps=2048,
                policy_kwargs=policy_kwargs,
                device=device,
            )

        # enable cudnn benchmark when using GPU
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        if user == 'up':
            timesteps = up_timesteps
        else:
            timesteps = down_timesteps

        model.learn(total_timesteps=timesteps)

        model.save(model_path)
        print(f"model_{user} saved")

        env.close()

        if quick:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a short quick training iteration for smoke test")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()
    main(quick=args.quick, n_envs=args.n_envs, force_cpu=args.cpu)
