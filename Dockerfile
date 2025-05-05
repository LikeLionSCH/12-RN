# Python 공식 이미지 사용
FROM python:3.8

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 코드 복사
COPY . /app

# 필요한 라이브러리 설치 (예: requirements.txt 사용)
RUN pip install gymnasium
RUN pip install sb3_contrib
RUN pip install FastAPI

# 컨테이너 실행 시 기본적으로 실행할 명령 설정
#CMD ["python", "app.py"]  # 메인 파일을 실행하도록 설정


