FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (using CPU-only torch to stay under 512MB RAM limit)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project files
COPY . .

# Ensure storage directories exist
RUN mkdir -p storage/chroma storage/bm25 logs data/raw data/processed

# Expose default Hugging Face Space port
EXPOSE 7860

# Run FastAPI server on port 7860
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]
