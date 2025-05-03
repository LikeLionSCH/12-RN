import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MiniChessEnv(gym.Env):
    """8×3 크기의 체스 유사 게임 환경 (차례 정보 포함)"""
    
    def __init__(self):
        super().__init__()
        
        # 보드 크기: 8×3, 기물 종류: 4개
        self.board_shape = (8, 3)
        self.num_pieces = 5 * 2
        
        # Observation: 보드와 현재 차례를 포함하는 딕셔너리 구조 사용
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.num_pieces, 8, 3), dtype=np.uint8),
            "turn": spaces.Discrete(2)  # 예: 0과 1로 플레이어 구분
        })
        
        # 행동 공간: 가능한 모든 출발지-도착지 (총 8*3*8*3 개)
        self.action_space = spaces.Discrete(8 * 3 * 8 * 3)
        self.time = 0
        self.reset()
    
    def reset(self, seed=None, options=None):
        """게임 초기화 및 초기 관측 반환"""
        super().reset(seed=seed)
        
        # 보드 초기화: 8×3×4 텐서 (각 기물별 채널)
        self.board = np.zeros((self.num_pieces, 8, 3), dtype=np.uint8)
        # 예제 초기 배치:
        self.board[0, 2, 0] = 1  # 상단 장 배치 / id: 0
        self.board[1, 2, 1] = 1  # 상단 왕 배치 / id: 1
        self.board[2, 2, 2] = 1  # 상단 상 배치 / id: 2
        self.board[3, 3, 1] = 1  # 상단 자 배치 / id: 3
        #상단 후 / id: 4
        
        self.board[5, 4, 1] = 1  # 하단 자 배치 / id: 5
        self.board[6, 5, 0] = 1  # 하단 상 배치 / id: 6
        self.board[7, 5, 1] = 1  # 하단 왕 배치 / id: 7
        self.board[8, 5, 2] = 1  # 상단 장 배치 / id: 8
        # 하단 후 / id: 9
        
        
        # 차례 정보 초기화 (0번 플레이어부터 시작)
        self.turn = 0
        self.time = 0
        
        return {"board": self.board, "turn": self.turn}, {}


    def step(self, action):
        """
        행동(action)은 0부터 8*3*8*3-1 까지의 정수라고 가정합니다.
        여기에서는 간단히 출발 위치와 도착 위치로 해석합니다.
        """
        # 예제: action을 출발지와 목표지 인덱스로 분할 (총 24칸)
        start_index = action // (8 * 3)
        target_index = action % (8 * 3)
        start_row, start_col = divmod(start_index, 3)
        target_row, target_col = divmod(target_index, 3)
        
        # 시작 위치에 현재 차례의 플레이어 소유 기물이 있는지 확인
        # (여기서는 단순화를 위해, board의 각 칸에는 단 하나의 기물만 있다고 가정)
        piece_found = False
        piece_id = None
        
        foundPid = self.get_piece_id(start_row, start_col)
        if foundPid > -1:
            piece_id = foundPid
            piece_found = True
        
        valid_move = False
        if piece_found:
            valid_move = self.validate_move(piece_id, (start_row, start_col), (target_row, target_col))
        
        reward = self.time * -1
        
        targetPid = self.get_piece_id(target_row, target_col)
        if valid_move and targetPid == 1:
            reward += 500
        elif valid_move and targetPid == 7:
            reward -= 500
        
        if targetPid != -1:
            self.board[targetPid, target_row, target_col] = 0

        # 이동 적용: 올바른 이동일 때만 기물을 옮기고 차례 변경
        if valid_move:
            self.board[piece_id, start_row, start_col] = 0
            self.board[piece_id, target_row, target_col] = 1
            self.turn = 1 - self.turn

        else:
            # 불법 이동 시, 차례 변경을 원하지 않을 수도 있음. (디자인에 따라 다름)
            reward = -1
        
        self.time += 1
        done = self.is_over()  # 종료조건
        
        return {"board": self.board, "turn": self.turn}, reward, done, False, {}
    
    def get_piece_id(self, row, col):
        for pid in range(self.num_pieces):
            if self.board[pid, row, col] == 1:
                return pid
        return -1


    def validate_move(self, piece_id, start_pos, target_pos):
        """기물별 이동 규칙 검증 """
        row_diff = abs(target_pos[0] - start_pos[0])
        col_diff = abs(target_pos[1] - start_pos[1])
        
        if target_pos[0] < 2 or target_pos[0] > 5: # 잡은 기물 칸으론 이둥 볼가
            return False
        
        if piece_id <= 4 and self.turn == 0:
            return False
        
        if piece_id >= 5 and self.turn == 1:
            return False
        
        targetPid = self.get_piece_id(target_pos[0], target_pos[1])
        if targetPid > -1 :
            if targetPid <= 4 and piece_id <= 4:
                return False
            elif targetPid >= 5 and piece_id >= 5:
                return False
        
        if piece_id == 0:  # id: 0 - 장 : 상하좌우 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1)
        elif piece_id == 1:  # id: 1 - 왕 : 상하좌우 대각선 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1) or \
                   (row_diff == 1 and col_diff == 1)
        elif piece_id == 2:  # id: 2 - 상 : 대각선 이동
            return (row_diff == 1 and col_diff == 1)
        elif piece_id == 3:  # id: 3 - 자 : 아래로 한칸 이동
            return (row_diff == 1 and col_diff == 0) and target_pos[0] > start_pos[0]
        elif piece_id == 4:  # id: 4 - 후 : 위로 대각 빼고 다 한칸 이동 가능
            return (row_diff == 1 and col_diff == 0) or (col_diff == 1 and row_diff == 0) or \
                   (row_diff == 1 and col_diff == 1 and target_pos[0] > start_pos[0])
                   
        elif piece_id == 5:  # id: 5 - 자 : 아래로 한칸 이동
            return (row_diff == 1 and col_diff == 0) and target_pos[0] < start_pos[0]
        elif piece_id == 6:  # id: 6 - 상 : 대각선 이동
            return (row_diff == 1 and col_diff == 1)
        elif piece_id == 7:  # id: 7 - 왕 : 상하좌우 대각선 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1) or \
                   (row_diff == 1 and col_diff == 1)
        elif piece_id == 8:  # id: 8 - 장 : 상하좌우 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1)
        elif piece_id == 9:  # id: 9 - 후 : 위로 대각 빼고 다 한칸 이동 가능
            return (row_diff == 1 and col_diff == 0) or (col_diff == 1 and row_diff == 0) or \
                   (row_diff == 1 and col_diff == 1 and target_pos[0] < start_pos[0])
        return False
    
    def is_over(self):
        if self.time > 500:
            return True
        king_white_id = 1  # 백 왕 ID (예제)
        king_black_id = 7  # 흑 왕 ID (예제)
    
        white_king_exists = (self.board[king_white_id] == 1).any()  # 백 왕이 있는지 확인
        black_king_exists = (self.board[king_black_id] == 1).any()  # 흑 왕이 있는지 확인
    
        # 둘 중 하나라도 사라졌으면 게임 종료
        return not (white_king_exists and black_king_exists) 
        
    def get_valid_actions(self):
        """현재 보드 상태 기준으로, 가능한 모든 행동(0~action_space.n-1)에 대해 유효성 마스크를 반환합니다."""
        valid_mask = np.zeros(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            start_index = action // (8 * 3)
            target_index = action % (8 * 3)
            start_row, start_col = divmod(start_index, 3)
            target_row, target_col = divmod(target_index, 3)
            
            piece_found = False
            piece_id = None
            for pid in range(self.num_pieces):
                if self.board[pid, start_row, start_col] == 1:
                    piece_found = True
                    piece_id = pid
                    break
            if piece_found and self.validate_move(piece_id, (start_row, start_col), (target_row, target_col)):
                valid_mask[action] = True
        return valid_mask
        
    
# 환경 등록 (이 환경은 __main__ 모듈에 정의되어 있다고 가정)
gym.register(
    id="MiniChess-v0",
    entry_point="__main__:MiniChessEnv"
)


from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


env = gym.make("MiniChess-v0")

env = ActionMasker(env, lambda env: env.unwrapped.get_valid_actions())
# MultiInputPolicy를 사용하여 모델 생성
model = MaskablePPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=10_000)
model.save("./12")


