import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from MiniChessEnv import MiniChessEnv

# 학습할 모델의 턴
user = 'up'  # up or down


for i in range(1, 100000):
    print("Training model_" + str(i))
    user = 'up' if i % 2 == 0 else 'down'

    # 상대 모델의 턴 설정
    enemy = 'down' if user == 'up' else 'up'
    
    # 환경 및 모델 생성
    env = gym.make("MiniChess-v0")
    env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
    
    custom_objects = {
        "clip_range": lambda _: 0.2,  # 기본값 설정
        "lr_schedule": lambda _: 2.5e-4  # 기본 학습률 설정
    }
    # 상대 모델 로드
    enemy_model = MaskablePPO.load("./model_" + str(enemy), env=env, custom_objects=custom_objects)
    env.unwrapped.set_enemy_env(enemy_model, enemy=0 if enemy == 'up' else 1)

    # 이전 모델을 로드하여 추가 학습 진행
    model_path = "./model_" + str(user)

    try:
        model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)
        model.verbose = 1
        model.ent_coef = 0.5
        model.gamma = 0.99  # 미래 보상 반영 증가
        model.learning_rate = 0.001 # 조금 더 적극적인 학습 하도록
        print(f"Loaded existing model: {model_path} (Continuing training)")
    except FileNotFoundError:
        print(f"No previous model found at {model_path}, training from scratch")
        model = MaskablePPO(
            "MultiInputPolicy", 
            env,
            verbose=1,
            batch_size=1024,  # 기존 유지
            learning_rate=0.01,
            ent_coef=0.5,  # 기존 0.2 → 0.5 (탐색 강화)
            n_steps=2048,  # 더 긴 학습 루프
            clip_range=0.2, # 기존 유지
            policy_kwargs={
               "net_arch": {
                   "pi": [256, 256, 256, 256],  # actor network
                   "vf": [256, 256, 256, 256]   # critic network
               }
           }
        )

    # 추가 학습
    if user == 'up':
        model.learn(total_timesteps=80_000)
    else :
        model.learn(total_timesteps=150_000)
        
    # 학습된 모델 저장
    model.save(model_path)
    
    print(f"model_{user} saved")
