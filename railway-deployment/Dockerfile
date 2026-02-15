FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# build-essential and cmake needed for compiling llama.cpp if wheels aren't found
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# NOTE: You must place the .gguf model in a 'models' directory relative to this Dockerfile
# Or mount it as a volume. For Railway, usually copying it into the image is easiest if < 2GB.
# COPY models/sbicard-slm.gguf /app/models/sbicard-slm.gguf

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
