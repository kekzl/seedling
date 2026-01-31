FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY config/ ./config/

# Create data directories
RUN mkdir -p /app/data /app/outputs

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "-m", "src.app"]
