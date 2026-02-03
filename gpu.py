import torch
import platform

def detect_device():
	# 우선 CUDA 확인
	if torch.cuda.is_available():
		return torch.device("cuda")
	# macOS MPS 확인
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	# 기본 CPU
	return torch.device("cpu")

device = detect_device()
print(f"Python platform: {platform.system()} {platform.release()}")
print(f"Torch version: {torch.__version__}")
print(f"Detected device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, "mps"):
	print(f"MPS built: {torch.backends.mps.is_built()}")
	print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
	try:
		print(f"CUDA device count: {torch.cuda.device_count()}")
		print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
	except Exception:
		pass
