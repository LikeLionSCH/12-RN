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
try:
    env = gym.make("MiniChess-v0")
    env.unwrapped.set_test_mode(True)  # 테스트 모드 활성화
    env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
except Exception as e:
    print(f"Environment init error: {e}")
    env = None

# MaskablePPO 모델 로드 (각 모델이 플레이어 차례에 따라 사용됨)
try:
    model_up = MaskablePPO.load("./models/model_up.zip")  # 예: 플레이어 0
    model_down = MaskablePPO.load("./models/model_down.zip")  # 예: 플레이어 1
except Exception as e:
    print(f"Model load error: {e}")
    model_up = None
    model_down = None

# 요청 바디 데이터 모델 정의
class InputObservation(BaseModel):
    # 실제 환경에서 사용되는 보드 배열은 (self.num_pieces, 8, 3)의 shape를 가져야 합니다.
    board: List[List[List[int]]]  # 예: 10 x 8 x 3 배열
    turn: int  # 현재 차례 (0 또는 1)
    is_new: bool = True  # 기본값 True로 설정

@app.get("/")
def read_root():
    return {"message": "MiniChess 예측 API 입니다."}



@app.post("/predict/")
def predict_next_move(data: InputObservation):
    """
    전달받은 보드 상태와 차례를 환경에 반영한 후,
    해당 차례에 맞는 모델로 예측하고 한 스텝 진행 후 결과를 반환합니다.
    """
    if model_up is None or model_down is None:
        return {"error": "Models not loaded"}
    if env is None:
        return {"error": "Environment not initialized"}
    
    # 먼저 reset을 호출하여 환경을 초기화함
    obs, info = env.reset()
    # API에서 받은 데이터를 환경에 적용
    env.unwrapped.board = np.array(data.board, dtype=np.uint8)
    env.unwrapped.turn = data.turn
    env.unwrapped.set_test_mode(True)
    
    # 캐시 무효화 (보드 변경 후 필수)
    env.unwrapped.two_d_board_num = -1
    
    # 유효한 행동 마스크 얻기 (turn 설정 후 반드시 갱신)
    action_mask = env.unwrapped.get_valid_actions()
    
    # 현재 차례에 따라 모델 선택 및 예측
    if data.turn == 0:
        # Turn 0: 상단 플레이어 (모델_up)
        obs_dict = {"board": env.unwrapped.board, "turn": data.turn}
        action, _states = model_up.predict(
            obs_dict,
            action_masks=action_mask,
            deterministic=True
        )
    elif data.turn == 1:
        # Turn 1: 하단 플레이어 (모델_down)
        obs_dict = {"board": env.unwrapped.board, "turn": data.turn}
        action, _states = model_down.predict(
            obs_dict,
            action_masks=action_mask,
            deterministic=True
        )
    else:
        return {"error": f"Invalid turn: {data.turn}"}
    
    # 행동 유효성 재검증
    if not action_mask[action]:
        return {"error": f"Model selected invalid action: {action}, turn: {data.turn}"}
    
    # 디버그: 선택한 행동의 출발/도착 위치 표시
    board_size = 8 * 3
    start_index = action // board_size
    target_index = action % board_size
    start_row, start_col = divmod(start_index, 3)
    target_row, target_col = divmod(target_index, 3)
    piece_id = env.unwrapped.get_piece_id(start_row, start_col)
    
    print(f"[DEBUG] board state before action:\n{env.unwrapped.board}")
    print(f"[DEBUG] turn={data.turn}, action={action}, piece_id={piece_id}, from=({start_row},{start_col}), to=({target_row},{target_col})")
    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
