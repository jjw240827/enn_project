FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Debian 패키지의 eng.traineddata는 속도 우선(fast) 모델이라, 인쇄체 정확도를 위해
# 더 무거운 대신 정확한 tessdata_best(LSTM) 버전으로 교체. 빌드 시에만 네트워크 필요,
# 런타임은 이미지에 내장돼 그대로 오프라인
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata', '/usr/share/tesseract-ocr/5/tessdata/eng.traineddata')"

WORKDIR /app

# GPU 없는 서버용 CPU 빌드 torch를 먼저 설치 (기본 PyPI 빌드는 CUDA 라이브러리 때문에 수 GB)
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 손글씨 인식 모델을 이미지에 미리 내려받아 두어 런타임에는 인터넷 없이(오프라인) 동작
ENV HANDWRITING_MODEL=microsoft/trocr-small-handwritten
RUN python -c "import os; from transformers import TrOCRProcessor, VisionEncoderDecoderModel; n=os.environ['HANDWRITING_MODEL']; TrOCRProcessor.from_pretrained(n); VisionEncoderDecoderModel.from_pretrained(n)"
ENV HF_HUB_OFFLINE=1

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
