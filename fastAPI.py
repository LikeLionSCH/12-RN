from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# MiniChessEnv 모듈에서 정의한 환경 임포트
from MiniChessEnv import MiniChessEnv

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 환경 생성 및 설정 (서버 시작 시 한 번 초기화)
env = gym.make("MiniChess-v0")
env.unwrapped.set_test_mode(True)  # 테스트 모드 활성화
env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())

# MaskablePPO 모델 로드 (각 모델이 플레이어 차례에 따라 사용됨)
model_up = MaskablePPO.load("./model_up.zip")  # 예: 플레이어 0
model_down = MaskablePPO.load("./model_down.zip")  # 예: 플레이어 1

# 요청 바디 데이터 모델 정의
class InputObservation(BaseModel):
    # 실제 환경에서 사용되는 보드 배열은 (self.num_pieces, 8, 3)의 shape를 가져야 합니다.
    board: List[List[List[int]]]  # 예: 10 x 8 x 3 배열
    turn: int  # 현재 차례 (0 또는 1)

@app.get("/")
def read_root():
    return {"message": "MiniChess 예측 API 입니다."}


# 먼저 reset을 호출하여 환경을 초기화함
obs, info = env.reset()

@app.post("/predict/")
def predict_next_move(data: InputObservation):
    """
    전달받은 보드 상태와 차례를 환경에 반영한 후,
    해당 차례에 맞는 모델로 예측하고 한 스텝 진행 후 결과를 반환합니다.
    """
    
    # API에서 받은 데이터를 환경에 적용
    env.unwrapped.board = np.array(data.board, dtype=np.uint8)
    env.unwrapped.turn = data.turn
    
    # 유효한 행동 마스크 얻기
    action_mask = env.unwrapped.get_valid_actions()
    
    # 현재 차례에 따라 모델 선택 및 예측
    if data.turn == 0:
        action, _states = model_up.predict(
            {"board": env.unwrapped.board, "turn": data.turn},
            action_masks=action_mask,
            deterministic=True
        )
    else:
        action, _states = model_down.predict(
            {"board": env.unwrapped.board, "turn": data.turn},
            action_masks=action_mask,
            deterministic=True
        )
    
    # 예측한 행동을 환경에 적용
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 예측 결과 반환
    return {
        "predicted_action": int(action),
        "reward": reward,
        "done": done,
        "observation": {
            "board": obs["board"].tolist(),  # numpy array를 JSON 직렬화 가능하도록 변환
            "turn": obs["turn"]
        },
        "info": info
    }

