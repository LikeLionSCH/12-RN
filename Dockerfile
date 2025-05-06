# Python 공식 이미지 사용
FROM python:3.12

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 코드 복사
COPY . /app

# 필요한 라이브러리 설치
RUN pip install gymnasium==1.0.0
RUN pip install sb3_contrib
RUN pip install FastAPI
RUN pip install uvicorn
RUN pip install numpy==2.2.0

# 컨테이너 실행 시 기본적으로 실행할 명령 설정
#CMD ["python", "learn.py"]  # 메인 파일을 실행하도록 설정

#uvicorn fastAPI:app --reload  --host 0.0.0.0