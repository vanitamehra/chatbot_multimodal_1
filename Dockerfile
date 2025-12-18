# ---------------------------
# 1. Base image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Environment variables
# ---------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------
# 3. Working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 4. Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    pkg-config \
    libcairo2-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 5. Copy requirements file
# ---------------------------
COPY requirements.txt .

# ---------------------------
# 6. Upgrade pip and install CPU Torch first
# ---------------------------
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1

# ---------------------------
# 7. Install remaining dependencies
# ---------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# 8. Copy application code
# ---------------------------
COPY . .

# Copy HuggingFace model into container
COPY models/flan_t5_base /app/models/flan_t5_base


# ---------------------------
# 9. Expose port
# ---------------------------
EXPOSE 8000

# ---------------------------
# 10. Run FastAPI app
# ---------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


