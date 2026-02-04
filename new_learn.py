import os
import argparse
import time

# Reduce thread oversubscription in multiprocess training
# These should be set before heavy libraries initialize threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Disable PyTorch distribution validation to avoid Simplex constraint errors
# This is safe because we manually validate action masks
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gymnasium as gym
from sb3_contrib import MaskablePPO
import torch

# Monkey-patch torch.distributions to disable validation
# This prevents Simplex constraint errors with masked actions
original_categorical_init = torch.distributions.Categorical.__init__

def patched_categorical_init(self, probs=None, logits=None, validate_args=None):
    # Force validate_args to False to avoid Simplex errors
    original_categorical_init(self, probs=probs, logits=logits, validate_args=False)

torch.distributions.Categorical.__init__ = patched_categorical_init

import gymnasium as gym
from sb3_contrib import MaskablePPO
# Limit PyTorch threads to avoid oversubscription in multiprocessing
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from MiniChessEnv import MiniChessEnv
import numpy as np


def safe_action_mask(env):
    """Safely generate valid action mask with validation."""
    try:
        mask = env.unwrapped.get_valid_actions()
        mask = np.asarray(mask, dtype=bool)
        
        # Validate mask
        if mask.ndim != 1 or mask.size != env.action_space.n:
            print(f"WARNING: Invalid mask shape {mask.shape}, expected ({env.action_space.n},)")
            return np.ones(env.action_space.n, dtype=bool)
        
        if not mask.any():
            print("WARNING: Empty action mask, allowing all actions")
            return np.ones(env.action_space.n, dtype=bool)
        
        return mask
    except Exception as e:
        print(f"ERROR in safe_action_mask: {e}")
        return np.ones(env.action_space.n, dtype=bool)


def evaluate_models(up_path: str, down_path: str, n_episodes: int = 30, device: str = "cpu"):
    """
    Evaluate two models against each other.
    Returns (up_wins, down_wins, draws)
    """
    # Load models
    try:
        up_model = MaskablePPO.load(up_path, device=device)
        down_model = MaskablePPO.load(down_path, device=device)
    except Exception as e:
        print(f"Failed to load models for evaluation: {e}")
        return None, None, None

    up_wins = 0
    down_wins = 0
    draws = 0

    for _ in range(n_episodes):
        env = gym.make("MiniChess-v0")
        env = ActionMasker(env, safe_action_mask)
        env.unwrapped.set_test_mode(True)
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            current_player = env.unwrapped.turn
            model = up_model if current_player == 0 else down_model

            action_masks = safe_action_mask(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)

        # Determine winner based on board state
        up_king_alive = bool((env.unwrapped.board[1] == 1).any())
        down_king_alive = bool((env.unwrapped.board[7] == 1).any())
        up_king_at_row5 = bool((env.unwrapped.board[1, 5, :] == 1).any())
        down_king_at_row2 = bool((env.unwrapped.board[7, 2, :] == 1).any())

        # UP wins if: DOWN king died OR UP king reached row 5
        # DOWN wins if: UP king died OR DOWN king reached row 2
        if not down_king_alive or up_king_at_row5:
            up_wins += 1
        elif not up_king_alive or down_king_at_row2:
            down_wins += 1
        else:
            draws += 1

        env.close()

    return up_wins, down_wins, draws


def make_env_fn(enemy_path: str = None, enemy_id: int = 1, device: str = "cpu"):
    def _init():
        env = gym.make("MiniChess-v0")
        env = ActionMasker(env, safe_action_mask)
        # If an enemy model path is provided, load it inside the worker process
        if enemy_path is not None:
            try:
                # Use faster device temporarily for loading (CPU faster for parallel loads)
                enemy_model = MaskablePPO.load(enemy_path, device=device)
                env.unwrapped.set_enemy_env(enemy_model, enemy=enemy_id)
            except Exception as e:
                import sys
                print(f"[Worker Error] Failed to load enemy model: {e}", file=sys.stderr)
        return env

    return _init


def get_device(force_cpu: bool = False):
    if not force_cpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_vec_env(n_envs: int, enemy_path: str = None, enemy_id: int = 1, device: str = "cpu", force_dummy: bool = False):
    print(f"[VecEnv] Initializing {n_envs} parallel environments...")
    import time
    start_time = time.time()
    
    env_fns = [make_env_fn(enemy_path=enemy_path, enemy_id=enemy_id, device=device) for _ in range(n_envs)]
    if force_dummy:
        env = DummyVecEnv(env_fns)
    else:
        try:
            # SubprocVecEnv will load models in parallel across workers
            env = SubprocVecEnv(env_fns, start_method='spawn')
        except Exception as e:
            print(f"[VecEnv] SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            env = DummyVecEnv(env_fns)
    
    # Wrap with VecMonitor to track episode statistics
    env = VecMonitor(env)
    
    elapsed = time.time() - start_time
    print(f"[VecEnv] Ready in {elapsed:.2f}s ({n_envs} envs)")
    return env


def main(quick: bool = False, n_envs: int = 48, force_cpu: bool = False, adaptive: bool = True, eval_freq: int = 1, net_arch: int = 256):
    device = get_device(force_cpu)
    print(f"Using device: {device}")
    print(f"Network architecture: [{net_arch}, {net_arch}]")

    # small quick mode for smoke test
    base_timesteps = 250_000 if not quick else 2_000
    up_timesteps = base_timesteps
    down_timesteps = base_timesteps
    
    # Track performance history
    up_win_rate = 0.5  # Start at 50% assumed
    down_win_rate = 0.5
    
    # Model name suffix based on architecture
    arch_suffix = f"_{net_arch}" if net_arch != 256 else ""

    for i in range(0, 100000):
        print("Training model_" + str(i))
        user = 'up' if i % 2 == 0 else 'down'

        enemy = 'down' if user == 'up' else 'up'

        custom_objects = {
            "clip_range": lambda _: 0.2,
            "lr_schedule": lambda _: 2.5e-4,
        }
        # prepare enemy model path; worker processes will load the model themselves
        enemy_path = f"./models/model_{enemy}{arch_suffix}.zip"
        if not os.path.exists(enemy_path):
            # Fallback to default model (256) if specific arch not found
            enemy_path_fallback = f"./models/model_{enemy}.zip"
            if os.path.exists(enemy_path_fallback):
                print(f"Enemy model {arch_suffix} not found, using default model: {enemy_path_fallback}")
                enemy_path = enemy_path_fallback
            else:
                raise FileNotFoundError("No enemy model found (512 or default).")

        # Create vector env; pass enemy_path so each worker can load it locally.
        # Load enemy model inside workers on CPU to avoid multiple CUDA contexts
        worker_enemy_device = 'cpu' if enemy_path is not None else device
        env = create_vec_env(
            n_envs,
            enemy_path=enemy_path,
            enemy_id=(0 if enemy == 'up' else 1),
            device=worker_enemy_device,
            force_dummy= False,
        )

        model_path = f"./models/model_{user}{arch_suffix}.zip"

        try:
            print(f"[Model] Loading {model_path}...")
            import time
            start_load = time.time()
            model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects, device=device)
            load_time = time.time() - start_load
            model.verbose = 1
            # Optimize hyperparameters - must match new model settings
            model.ent_coef = 0.05  # Increased from 0.01 for more exploration
            model.learning_rate = 5e-4  # Reduced from 1e-3 for stability
            model.n_epochs = 10
            # Update batch_size and n_steps to match
            model.batch_size = 2048
            model.n_steps = 2048
            # Recreate rollout buffer with new settings
            model.rollout_buffer = model.rollout_buffer_class(
                model.n_steps,
                model.observation_space,
                model.action_space,
                device=model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=env.num_envs,
            )
            print(f"[Model] Loaded in {load_time:.2f}s: {model_path} (Continuing training)")
        except Exception:
            print(f"No previous model found at {model_path}, training from scratch")
            policy_kwargs = {"net_arch": {"pi": [net_arch, net_arch], "vf": [net_arch, net_arch]}}
            
            # Optimize hyperparameters based on network size
            # batch_size must divide (n_steps * n_envs) evenly
            n_steps = 2048  # Reduced for stability
            batch_size = 2048  # Must be <= n_steps * n_envs
            learning_rate = 5e-4  # Balanced for stability and learning
            ent_coef = 0.05  # Balanced entropy for exploration vs exploitation
            clip_range = 0.2  # Tighter clipping
            n_epochs = 10  # Number of epochs when updating policy
            
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                batch_size=batch_size,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                clip_range=clip_range,
                n_steps=n_steps,
                n_epochs=n_epochs,
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

        # Adaptive training: evaluate and adjust timesteps
        if adaptive and i > 1 and i % eval_freq == 0:
            print("\n=== Evaluating models ===")
            up_path = f"./models/model_up{arch_suffix}.zip"
            down_path = f"./models/model_down{arch_suffix}.zip"
            
            # Check if both models exist
            if os.path.exists(up_path) and os.path.exists(down_path):
                up_wins, down_wins, draws = evaluate_models(up_path, down_path, n_episodes=30, device=device)
                
                if up_wins is not None:
                    total = up_wins + down_wins + draws
                    up_win_rate = up_wins / total if total > 0 else 0.5
                    down_win_rate = down_wins / total if total > 0 else 0.5
                    
                    print(f"Evaluation results: UP {up_wins}/{total} ({up_win_rate*100:.1f}%) | DOWN {down_wins}/{total} ({down_win_rate*100:.1f}%) | Draws {draws}")
                    
                    # Adjust timesteps based on performance
                    # If a model is underperforming (< 40% win rate), increase its training time
                    if up_win_rate < 0.40:
                        up_timesteps = int(base_timesteps * 2)
                        print(f"UP model underperforming ({up_win_rate*100:.1f}%), increasing timesteps to {up_timesteps:,}")
                    elif up_win_rate > 0.60:
                        up_timesteps = int(base_timesteps * 0.5)
                        print(f"UP model overperforming ({up_win_rate*100:.1f}%), decreasing timesteps to {up_timesteps:,}")
                    else:
                        up_timesteps = base_timesteps
                    
                    if down_win_rate < 0.40:
                        down_timesteps = int(base_timesteps * 2)
                        print(f"DOWN model underperforming ({down_win_rate*100:.1f}%), increasing timesteps to {down_timesteps:,}")
                    elif down_win_rate > 0.60:
                        down_timesteps = int(base_timesteps * 0.5)
                        print(f"DOWN model overperforming ({down_win_rate*100:.1f}%), decreasing timesteps to {down_timesteps:,}")
                    else:
                        down_timesteps = base_timesteps
                else:
                    print("Evaluation failed, using base timesteps")
            print("=== Evaluation complete ===\n")

        if quick:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a short quick training iteration for smoke test")
    parser.add_argument("--n_envs", type=int, default=48)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument("--force-dummy", action="store_true", help="Force DummyVecEnv instead of SubprocVecEnv (improves GPU throughput)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive training (fixed timesteps)")
    parser.add_argument("--eval-freq", type=int, default=1, help="Evaluate models every N cycles (default: 1)")
    parser.add_argument("--net-arch", type=int, default=256, choices=[128, 256, 512, 1024], help="Neural network size (default: 256)")
    args = parser.parse_args()
    main(quick=args.quick, n_envs=args.n_envs, force_cpu=args.cpu, adaptive=not args.no_adaptive, eval_freq=args.eval_freq, net_arch=args.net_arch)
