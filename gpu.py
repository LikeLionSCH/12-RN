import torch
device = torch.device("mps")
print(f"현재 사용 중인 장치: {device}")
print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")
