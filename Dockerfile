FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
# default: 1-minute synthetic smoke run (no download). Full run: python scripts/prepare_data.py && python scripts/replay_eval.py
CMD ["python", "scripts/replay_eval.py", "--synthetic", "--out", "results/replay_synth.json"]
