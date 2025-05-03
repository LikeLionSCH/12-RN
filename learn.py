import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from MiniChessEnv import MiniChessEnv

# 학습할 모델의 턴
user = 1  # 1 or 0

# 환경 및 모델 생성

env = gym.make("MiniChess-v0")
env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
embedded_model = MaskablePPO.load("./new2_0", env=env)
env.unwrapped.set_embedded_env(env, embedded_model, user)
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("./new2_" + str(user))


