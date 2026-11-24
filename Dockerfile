FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. Install lightweight CPU-only PyTorch first (saves 2+ GB of download)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining core dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application files
COPY . .

EXPOSE 8000

# 4. Ingest on startup if not present, then launch API
CMD sh -c "python ingest.py && uvicorn main:app --host 0.0.0.0 --port 8000"