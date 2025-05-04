import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from MiniChessEnv import MiniChessEnv
    
# 학습할 모델의 턴
user = 'up'  # up or down

for i in range(1, 100):
    print("Training model_" + str(i))
    user = 'up' if i % 2 == 0 else 'down'

    # 상대 모델의 턴 설정
    enemy = 'down' if user == 'up' else 'up'
    
    # 환경 및 모델 생성
    env = gym.make("MiniChess-v0")
    env.reset()
    env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
    embedded_model = MaskablePPO.load("./model_" + str(enemy), env=env)
    env.unwrapped.set_embedded_env(env, embedded_model, enemy = 0 if enemy == 'up' else 1)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, batch_size=256, learning_rate=0.03 / i, 
                        ent_coef=0.2)
    model.learn(total_timesteps=10_000)
    model.save("./model_" + str(user))
    
    print("model_" + str(user) + " saved")