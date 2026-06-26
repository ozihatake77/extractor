FROM python:3.11-slim

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models during build
RUN python3 -c "import easyocr; easyocr.Reader(['id', 'en'], gpu=False)"

# Copy app
COPY . .

# Expose port
EXPOSE ${PORT:-5000}

# Run
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300
