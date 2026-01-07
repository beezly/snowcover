FROM python:3.11-slim

# Install OpenCV dependencies and ffmpeg for RTSP handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 snowcover

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with correct ownership
COPY --chown=snowcover:snowcover src/ ./src/
COPY --chown=snowcover:snowcover models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

USER snowcover

# Run the application
ENTRYPOINT ["python", "-m", "snowcover.main"]
CMD []
