FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Model, Dual Scalers, Baseline, and App Files
COPY models/strain_vision_net.onnx ./models/
COPY artifacts/spatial_scaler.pkl ./artifacts/
COPY artifacts/tabular_scaler.pkl ./artifacts/
COPY artifacts/strain_drift_baseline.json ./artifacts/
COPY config.py app.py drift.py inference.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]