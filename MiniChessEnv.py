import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 환경 등록 (이 환경은 __main__ 모듈에 정의되어 있다고 가정)
gym.register(
    id="MiniChess-v0",
    entry_point="MiniChessEnv:MiniChessEnv"
)

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
        self.is_test_mode = False

        # 학습시 상대 모델 적용
        self.embedded_env = None
        self.embedded_model = None
        self.user = None

        self.reset()
        
    def set_test_mode(self, is_test_mode):
        """테스트 모드 설정"""
        self.is_test_mode = is_test_mode

    def set_embedded_env(self, env, model, user):
        """상대 모델 설정"""
        self.embedded_env = env
        self.embedded_model = model
        self.user = user
    
    
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
        # 상단 후 / id: 4
        
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
        start_index = action // (8 * 3)
        target_index = action % (8 * 3)
        start_row, start_col = divmod(start_index, 3)
        target_row, target_col = divmod(target_index, 3)
        
        piece_found = False
        piece_id = None
        
        foundPid = self.get_piece_id(start_row, start_col)
        if foundPid > -1:
            piece_id = foundPid
            piece_found = True
        
        valid_move = False
        if piece_found:
            valid_move = self.validate_move(piece_id, (start_row, start_col), (target_row, target_col))
        
        if self.is_test_mode:
            print("action:", action)
            print(start_row, start_col, "에서", target_row, target_col, "로 이동")
            print("기물:", self.get_kr_from_pid(piece_id))

        reward = self.time * -1
        
        targetPid = self.get_piece_id(target_row, target_col)
        if valid_move and targetPid == 1:
            reward += 500 if self.user == "down" else -500
        elif valid_move and targetPid == 7:
            reward += -500 if self.user == 'down' else 500
        
        # 잡은 기물이 있음
        if targetPid != -1:
            self.catch_piece(targetPid)


        # 이동 적용
        self.board[piece_id, start_row, start_col] = 0
        self.board[piece_id, target_row, target_col] = 1
        self.turn = 1 - self.turn

        obs = {"board": self.board, "turn": self.turn}

        # 상대 기물 이동
        if not self.is_test_mode and self.turn == self.user :
            action_mask = self.embedded_env.unwrapped.get_valid_actions()
            action, _states = self.embedded_model.predict({"board": self.board, "turn": self.turn}, action_masks=action_mask, deterministic=False)
            obs, reward, terminated, truncated, info = self.embedded_env.step(action)
            self.turn = 1 - self.turn
        self.time += 1

        done = self.is_over()  # 종료조건
        
        return obs, reward, done, False, {}
    
    def get_piece_id(self, row, col):
        for pid in range(self.num_pieces):
            if self.board[pid, row, col] == 1:
                return pid
        return -1
    
    def catch_piece(self, targetPid):
        #윗 장을 잡음
        if targetPid == 0:
            if self.board[8, 6, 1] == 1:
                self.board[8, 7, 1] = 1
            else:
                self.board[8, 6, 1] = 1
        #윗 상을 잡음
        elif targetPid == 2:
            if self.board[6, 6, 2] == 1:
                self.board[6, 7, 2] = 1
            else:
                self.board[6, 6, 2] = 1
        #윗 자를 잡음
        elif targetPid == 3:
            if self.board[5, 6, 0] == 1:
                self.board[5, 7, 0] = 1
            else:
                self.board[5, 6, 0] = 1
        #윗 후를 잡음
        elif targetPid == 4:
            if self.board[5, 6, 0] == 1:
                self.board[5, 7, 0] = 1
            else:
                self.board[5, 6, 0] = 1

        #아랫 장을 잡음
        elif targetPid == 8:
            if self.board[0, 1, 1] == 1:
                self.board[0, 0, 1] = 1
            else:
                self.board[0, 1, 1] = 1
        #아랫 상을 잡음
        elif targetPid == 6:
            if self.board[2, 1, 2] == 1:
                self.board[2, 0, 2] = 1
            else:
                self.board[2, 1, 2] = 1
        #아랫 자를 잡음
        elif targetPid == 5:
            if self.board[3, 1, 0] == 1:
                self.board[3, 0, 0] = 1
            else:
                self.board[3, 1, 0] = 1
        #아랫 후를 잡음
        elif targetPid == 9:
            if self.board[3, 1, 0] == 1:
                self.board[3, 0, 0] = 1
            else:
                self.board[3, 1, 0] = 1

    def validate_move(self, piece_id, start_pos, target_pos):
        """기물별 이동 규칙 검증 """
        row_diff = abs(target_pos[0] - start_pos[0])
        col_diff = abs(target_pos[1] - start_pos[1])
        
        #게임 안에 있는 기물은 게임 안에서민 이동해야 함
        if start_pos[0] >= 2 and start_pos[0] <= 5:
            if target_pos[0] < 2 or target_pos[0] > 5:
                return False
            
        #윗사람이 잡은 기물 놓기
        if start_pos[0] < 2:
            if target_pos[0] >= 5:
                return False
            return self.get_piece_id(target_pos[0], target_pos[1]) == -1
        
        #아랫사람이 잡은 기물 놓기
        elif start_pos[0] > 5:
            if target_pos[0] <= 2:
                return False
            return self.get_piece_id(target_pos[0], target_pos[1]) == -1
        
        if piece_id <= 4 and self.turn == 1:
            return False
        
        if piece_id >= 5 and self.turn == 0:
            return False
        
        targetPid = self.get_piece_id(target_pos[0], target_pos[1])
        if targetPid > -1:
            if targetPid <= 4 and piece_id <= 4:
                return False
            elif targetPid >= 5 and piece_id >= 5:
                return False
        
        if piece_id == 0:  # 장: 상하좌우 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1)
        elif piece_id == 1:  # 왕: 상하좌우 대각선 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1) or (row_diff == 1 and col_diff == 1)
        elif piece_id == 2:  # 상: 대각선 이동
            return (row_diff == 1 and col_diff == 1)
        elif piece_id == 3:  # 자: 아래로 한 칸 이동
            return (row_diff == 1 and col_diff == 0) and target_pos[0] > start_pos[0]
        elif piece_id == 4:  # 후: 위로 대각 빼고 한 칸 이동
            return (row_diff == 1 and col_diff == 0) or (col_diff == 1 and row_diff == 0) or (row_diff == 1 and col_diff == 1 and target_pos[0] > start_pos[0])
        elif piece_id == 5:  # 자: 아래로 한 칸 이동
            return (row_diff == 1 and col_diff == 0) and target_pos[0] < start_pos[0]
        elif piece_id == 6:  # 상: 대각선 이동
            return (row_diff == 1 and col_diff == 1)
        elif piece_id == 7:  # 왕: 상하좌우 대각선 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1) or (row_diff == 1 and col_diff == 1)
        elif piece_id == 8:  # 장: 상하좌우 이동
            return (row_diff == 0 and col_diff == 1) or (col_diff == 0 and row_diff == 1)
        elif piece_id == 9:  # 후: 위로 대각 빼고 한 칸 이동
            return (row_diff == 1 and col_diff == 0) or (col_diff == 1 and row_diff == 0) or (row_diff == 1 and col_diff == 1 and target_pos[0] < start_pos[0])
        return False
    
    def is_over(self):
        if self.time > 500:
            return True
        king_white_id = 1  # 백 왕 ID (예제)
        king_black_id = 7  # 흑 왕 ID (예제)
    
        white_king_exists = (self.board[king_white_id] == 1).any()
        black_king_exists = (self.board[king_black_id] == 1).any()
    
        return not (white_king_exists and black_king_exists) 
        
    def get_valid_actions(self):
        """현재 보드 상태 기준으로 가능한 모든 행동에 대해 유효성 마스크를 반환합니다."""
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

    def get_kr_from_pid(self, pid):
        if pid == 0:
            return "장(상)"
        elif pid == 1:
            return "왕(상)"
        elif pid == 2:
            return "상(상)"
        elif pid == 3:
            return "자(상)"
        elif pid == 4:
            return "후(상)"
        elif pid == 5:
            return "자(하)"
        elif pid == 6:
            return "상(하)"
        elif pid == 7:
            return "왕(하)"
        elif pid == 8:
            return "장(하)"
        elif pid == 9:
            return "후(하)"
        else:
            return None