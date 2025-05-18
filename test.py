from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from MiniChessEnv import MiniChessEnv

# 환경 생성
env = gym.make("MiniChess-v0")

env.unwrapped.set_test_mode(True)  # 테스트 모드 설정

# 원본 환경에서 `get_valid_actions()`를 올바르게 참조하도록 래퍼 적용
env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())


# MaskablePPO 모델 불러오기
model_up = MaskablePPO.load("./new_model_up")
model_down = MaskablePPO.load("./new_model_down")

## 모델 테스트
num_episodes = 10  # 테스트할 에피소드 수

# 승리 횟수 추적 변수 추가
wins_model_up = 0
wins_model_down = 0
draws = 0

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    done = False
    episode_reward = 0

    while not done:
        # ✅ 원본 환경(`unwrapped`)에서 `get_valid_actions()` 호출
        action_mask = env.unwrapped.get_valid_actions()
        
        # 모델 실행 시 마스킹 적용 (유효한 행동만 선택)
        if obs["turn"] == 0:
            action, _states = model_up.predict(obs, action_masks=action_mask, deterministic=False)
        else:
            action, _states = model_down.predict(obs, action_masks=action_mask, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    # 승리 판정
    if episode_reward > 0:
        wins_model_up += 1
    elif episode_reward < 0:
        wins_model_down += 1
    else:
        draws += 1

    print(f"Episode {ep+1}: Total Reward = {episode_reward}")

# 승률 출력
print(f"Model up Wins: {wins_model_up} ({wins_model_up / num_episodes * 100:.2f}%)")
print(f"Model down Wins: {wins_model_down} ({wins_model_down / num_episodes * 100:.2f}%)")
print(f"Draws: {draws} ({draws / num_episodes * 100:.2f}%)")

env.close()
