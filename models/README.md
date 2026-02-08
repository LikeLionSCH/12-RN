---
tags:
- reinforcement-learning
- maskable-ppo
- korean-chess
- 12janggi
- self-play
- stable-baselines3
- sb3-contrib
library_name: stable-baselines3
license: mit
language:
- ko
---

# 🎮 십이장기 (12-Janggi) Reinforcement Learning Models

Deep reinforcement learning agents trained to play 십이장기, a simplified Korean chess variant played on an 8×3 board, using self-play and Maskable PPO algorithm.

## 📋 Model Overview

| Property | Value |
|----------|-------|
| **Algorithm** | Maskable PPO (Proximal Policy Optimization) |
| **Framework** | Stable-Baselines3-Contrib |
| **Training Method** | Self-play with adaptive timesteps |
| **Board Dimensions** | 8 rows × 3 columns |
| **Total Actions** | 576 (24×24 position combinations) |
| **Game Type** | 십이장기 (Korean Chess Variant) |

## 🎯 십이장기 Game Rules

**Pieces (기물):**
- 🤴 **왕 (King)**: Moves one step in any direction (horizontal, vertical, diagonal)
- 💎 **상 (Advisor/Elephant)**: Moves one step diagonally
- 🐘 **장 (General)**: Moves one step horizontally or vertically
- 🚗 **자 (Chariot/Soldier)**: Moves one step forward (promotes to 후 at end row)
- ⭐ **후 (Promoted Chariot)**: Enhanced movement after promotion

**Victory Conditions:**
- Capture opponent's King (왕)
- Move your King to opponent's back row (UP: row 5, DOWN: row 2)

**Game Features:**
- Simplified Korean chess with only 12 pieces (6 per player)
- Fast-paced gameplay on compact 8×3 board
- Captured pieces can be placed back on the board

## 📦 Available Models

- **`model_down.zip`**: Standard (256 neurons) - DOWN player (下家)
- **`model_down_512.zip`**: Large (512 neurons) - DOWN player (下家)
- **`model_up.zip`**: Standard (256 neurons) - UP player (上家)
- **`model_up_512.zip`**: Large (512 neurons) - UP player (上家)

## 🚀 Quick Start

### Installation

```bash
pip install sb3-contrib gymnasium numpy
```

### Load and Use Model

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym

# Load 십이장기 environment
env = gym.make("MiniChess-v0")  # Environment name kept for compatibility
env = ActionMasker(env, lambda e: e.unwrapped.get_valid_actions())

# Load trained model
model = MaskablePPO.load("model_up.zip")

# Play
obs, info = env.reset()
done = False

while not done:
    # Get valid actions
    action_masks = env.unwrapped.get_valid_actions()
    
    # Predict with masking
    action, _states = model.predict(obs, action_masks=action_masks, deterministic=False)
    
    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Game ended with reward: {reward}")
```

### Download from Hub

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="SoonchunhyangUniversity/12-RN",
    filename="model_up.zip"
)

model = MaskablePPO.load(model_path)
```

## 🎓 Training Details

### Hyperparameters

- **Learning Rate**: 5e-4 with adaptive adjustment
- **Batch Size**: 2048
- **N-Steps**: 2048
- **Entropy Coefficient**: 0.05 (exploration vs exploitation balance)
- **Clip Range**: 0.2
- **N-Epochs**: 10
- **Network Architecture**: [256, 256] or [512, 512] (policy and value networks)

### Training Strategy

- **Self-play**: Models alternate training against each other
- **Adaptive Training**: Timesteps adjusted based on win rate performance
- **Action Masking**: Only legal moves are considered during training
- **Parallel Environments**: 48 simultaneous game instances for efficient data collection

### Observation Space

Multi-channel board representation (Dict):
- **Board**: `(10, 8, 3)` tensor - 10 channels for different piece types and players
- **Turn**: Scalar indicating current player (0 or 1)

### Action Space

- **Type**: Discrete(576)
- **Encoding**: `action = start_position * 24 + target_position`
- **Masking**: Invalid moves filtered via action masks

## 📊 Performance

Models achieve strong performance through self-play:
- Average episode reward: ~200-400 range
- Episode length: 15-25 moves on average
- Win rate stabilizes around 45-55% when evenly matched

## 🔗 Resources

- **GitHub Repository**: [LikeLionSCH/12-RN](https://github.com/LikeLionSCH/12-RN)
- **Environment Code**: See repository for `MiniChessEnv` implementation
- **Training Script**: `new_learn.py` in repository

## 📄 Citation

If you use these models in your research, please cite:

```bibtex
@misc{12janggi_rl_2026,
  title={십이장기 (12-Janggi) Reinforcement Learning Models},
  author={Soonchunhyang University},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/SoonchunhyangUniversity/12-RN}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Developed by**: Soonchunhyang University (순천향대학교)  
**Contact**: For questions or collaborations, please open an issue on GitHub
