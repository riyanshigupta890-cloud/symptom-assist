FROM python:3.10-slim
# Install system utilities needed for compile-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app
# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application files
COPY . .
# Pre-create writable folders
RUN mkdir -p /app/logs /app/data/chroma_db && \
    chmod -R 777 /app/logs /app/data
# Render uses port 10000 by default
EXPOSE 10000
# Start Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]