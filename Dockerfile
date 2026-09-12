FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY frontend ./frontend
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY data ./data

RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
