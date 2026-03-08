FROM python:3.11-slim

WORKDIR /app

# Minimal system libs required by OpenCV headless + PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py processor.py pdf_generator.py ./
COPY static/ static/

# Render / Railway / Fly.io all inject PORT at runtime
ENV PORT=8080

EXPOSE 8080

# 1 worker keeps memory low on free-tier containers;
# 180 s timeout handles large scanned PDFs
CMD gunicorn --bind "0.0.0.0:${PORT}" --timeout 180 --workers 1 app:app
