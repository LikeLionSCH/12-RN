import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from MiniChessEnv import MiniChessEnv

# 학습할 모델의 턴
user = 'down'  # up or down

# 상대 모델의 턴 설정
enemy = 'down' if user == 'up' else 'up'

# 환경 및 모델 생성
env = gym.make("MiniChess-v0")
env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
embedded_model = MaskablePPO.load("./model_" + str(enemy), env=env)
env.unwrapped.set_embedded_env(env, embedded_model, user = 0 if user == 'up' else 1)
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("./model_" + str(user))

print("model_" + str(user) + " saved")