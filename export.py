import torch
import onnx
from sb3_contrib import MaskablePPO

# 학습된 모델 로드
model = MaskablePPO.load("new2_1.zip")

# MiniChess 환경에서 observation_space는 딕셔너리 형태로 되어 있습니다.
# "board"의 크기는 (10, 8, 3)이며, "turn"은 Discrete(2)에 해당합니다.
dummy_board = torch.randn(1, 10, 8, 3)  # "board" 입력: (batch_size, num_pieces, 8, 3)
dummy_turn = torch.tensor([0], dtype=torch.int64)  # "turn" 입력: (batch_size,)

# PolicyWrapper 모듈: policy가 딕셔너리 입력을 요구하므로, 입력 텐서들을 dict로 구성
# 그리고 forward() 호출 결과 중 행동(action)값만 반환하도록 합니다.
class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super(PolicyWrapper, self).__init__()
        self.policy = policy

    def forward(self, board, turn):
        obs = {"board": board, "turn": turn}
        # self.policy.forward(obs) 가 반환하는 값은 여러 개의 요소를 가진 튜플일 수 있습니다.
        # 여기서는 첫 번째 요소(예: 행동)를 사용한다고 가정합니다.
        outputs = self.policy.forward(obs)
        action = outputs[0]
        logits = outputs[1]
        return action, logits
# Wrapper 생성
wrapper = PolicyWrapper(model.policy)

# ONNX 변환: wrapper.forward는 두 개의 인자(board, turn)를 받도록 작성됨.
torch.onnx.export(
    wrapper,
    (dummy_board, dummy_turn),
    "new2_1.onnx",
    input_names=["board", "turn"],
    output_names=["action", "logits"],
    opset_version=11
)

print("ONNX 모델이 'new2_1.onnx'로 저장되었습니다.")
