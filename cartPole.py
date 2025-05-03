import gymnasium as gym  # Gymnasium 라이브러리 불러오기 (강화학습 환경 제공)
import numpy as np  # 숫자 계산을 위한 NumPy
import torch  # PyTorch를 사용하여 신경망 구현
import torch.nn as nn  # 신경망 구성 요소 불러오기
import torch.optim as optim  # 최적화 알고리즘 제공
import random  # 랜덤 값 생성
import matplotlib.pyplot as plt  # 학습 과정 시각화를 위한 Matplotlib
from collections import deque  # 경험 재생 메모리 (Replay Buffer) 구현

# 💡 환경 생성: 'CartPole-v1'은 막대기를 균형 있게 유지하는 RL 문제
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # 상태(State) 공간 크기 (4차원)
action_dim = env.action_space.n  # 행동(Action) 공간 크기 (2차원: 왼쪽/오른쪽)

# 🎯 Q-Network 정의 (DQN의 신경망)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # 첫 번째 은닉층 (128개 뉴런)
        self.fc2 = nn.Linear(128, 128)  # 두 번째 은닉층 (128개 뉴런)
        self.fc3 = nn.Linear(128, action_dim)  # 출력층 (행동의 Q값 출력)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 첫 번째 층 활성화 함수 (ReLU)
        x = torch.relu(self.fc2(x))  # 두 번째 층 활성화 함수 (ReLU)
        return self.fc3(x)  # 행동(Action)에 대한 Q값 출력

# 🔧 하이퍼파라미터 설정 (학습 속도, 탐색률 등)
learning_rate = 0.001  # 학습률 (Learning Rate)
gamma = 0.99  # 할인율 (Discount Factor)
epsilon = 1.0  # 초기 탐색률 (Exploration Rate)
epsilon_decay = 0.995  # 탐색률 감소율 (Epsilon Decay)
epsilon_min = 0.01  # 최소 탐색률
batch_size = 64  # 미니배치 크기
memory_size = 10000  # 경험 재생 메모리 크기
target_update = 10  # 타겟 네트워크 업데이트 주기
episodes = 500  # 총 학습 에피소드 수

# 🚀 모델 및 옵티마이저 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부 확인
q_network = QNetwork(state_dim, action_dim).to(device)  # 현재 학습 네트워크
target_network = QNetwork(state_dim, action_dim).to(device)  # 타겟 네트워크
target_network.load_state_dict(q_network.state_dict())  # 초기 타겟 네트워크 동기화
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)  # Adam 옵티마이저 설정

# 🔄 경험 재생 메모리 (Replay Buffer) 생성
memory = deque(maxlen=memory_size)  # 최대 크기 10,000으로 설정

# 🎭 행동 선택 함수 (Epsilon-Greedy Policy)
def select_action(state, epsilon):
    if random.random() < epsilon:  # 탐색: 랜덤 행동 선택
        return env.action_space.sample()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 상태 변환
    with torch.no_grad():  # 학습 없이 Q값 계산
        q_values = q_network(state_tensor)
    return torch.argmax(q_values).item()  # 가장 높은 Q값을 가진 행동 선택

# 🏋️‍♂️ 학습 함수 (DQN 업데이트)
def train():
    if len(memory) < batch_size:  # 미니배치 크기보다 데이터가 부족하면 학습 X
        return

    batch = random.sample(memory, batch_size)  # 메모리에서 배치 크기만큼 샘플링
    states, actions, rewards, next_states, dones = zip(*batch)  # 배치 데이터 분리

    # Tensor 변환 및 GPU 할당
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # 🎯 현재 Q값 계산
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_network(next_states).max(1)[0]  # 타겟 네트워크 Q값
    target_q_values = rewards + gamma * next_q_values * (1 - dones)  # 업데이트 공식

    # 손실(Loss) 계산 및 최적화 수행
    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 📊 학습 결과 저장용 리스트
episode_rewards = []

# 🔥 학습 실행
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(200):  # 한 에피소드당 최대 200 스텝
        action = select_action(state, epsilon)  # 행동 선택
        next_state, reward, done, _, _ = env.step(action)  # 환경 진행
        
        memory.append((state, action, reward, next_state, done))  # 경험 저장
        state = next_state  # 상태 업데이트
        total_reward += reward

        train()  # 학습 수행

        if done:
            break  # 환경 종료 시 탈출

    epsilon = max(epsilon * epsilon_decay, epsilon_min)  # 탐색률 감소
    episode_rewards.append(total_reward)  # 보상 저장

    if episode % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())  # 타겟 네트워크 업데이트

    print(f"Episode {episode}: Total Reward = {total_reward}")

env.close()

# 📈 학습 과정 시각화 (Total Reward 그래프)
plt.plot(range(episodes), episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reinforcement Learning Training Progress")
plt.show()

# 🎥 학습된 에이전트 실행 (환경 시각화)
env = gym.make("CartPole-v1", render_mode="human")  # 인간이 볼 수 있도록 시각화 모드 활성화
for episode in range(5):  # 5개 에피소드 실행하여 행동 확인
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(200):
        env.render()  # 환경 화면 출력
        action = select_action(state, 0)  # 학습된 행동 사용 (탐색 없이 결정론적 선택)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            break  # 에피소드 종료 시 탈출

    print(f"Test Episode {episode}: Total Reward = {total_reward}")

env.close()
