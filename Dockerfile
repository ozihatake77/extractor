FROM python:3.11-slim

# Install tesseract + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models
RUN python3 -c "import easyocr; easyocr.Reader(['en','id'], gpu=False, verbose=False)" 2>/dev/null || true

COPY . .

EXPOSE ${PORT:-5000}

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 180
