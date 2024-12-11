# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create home directory for non-root user and set permissions
RUN mkdir -p /home/appuser && \
    chmod 777 /home/appuser

# Set HOME environment variable
ENV HOME=/home/appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and clean up in the same layer
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copy the rest of the application
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Set default command
CMD ["python", "src/main.py", "-d"] 