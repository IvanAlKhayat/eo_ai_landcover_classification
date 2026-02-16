# Multi-stage build for optimized production image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY models/ ./models/
COPY data/preprocess.py ./data/
COPY inference.py .
COPY api_server.py .

# Create directories for data
RUN mkdir -p /data/input /data/output checkpoints

# Copy model checkpoint (you'll need to add this)
# COPY checkpoints/quantized_model.pth ./checkpoints/

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run API server
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# BUILD AND RUN INSTRUCTIONS:
# ============================================================================
#
# 1. Build the image:
#    docker build -t eo-inference:latest .
#
# 2. Run the container:
#    docker run -d -p 8000:8000 \
#      -v $(pwd)/data:/data \
#      -v $(pwd)/checkpoints:/app/checkpoints \
#      --name eo-api \
#      eo-inference:latest
#
# 3. Test the API:
#    curl http://localhost:8000/health
#    curl -X POST http://localhost:8000/predict \
#      -F "image=@sample.npy" \
#      -o prediction.png
#
# 4. View logs:
#    docker logs eo-api
#
# 5. Stop and remove:
#    docker stop eo-api
#    docker rm eo-api
#
# ============================================================================
# DEPLOYMENT OPTIONS:
# ============================================================================
#
# AWS ECS:
#   - Push to ECR: aws ecr get-login-password | docker login ...
#   - Create task definition with GPU support (if needed)
#   - Deploy to ECS cluster
#
# Kubernetes:
#   - Create deployment.yaml with resource limits
#   - Use HPA for auto-scaling
#   - Mount PVC for model storage
#
# Cloud Run (GCP):
#   - gcloud run deploy eo-api --image gcr.io/project/eo-inference
#   - Set memory and CPU limits
#   - Configure autoscaling
#
# ============================================================================
