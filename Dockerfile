FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Force correct wheels and explicitly install python-multipart
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && python -m pip install --no-cache-dir python-multipart \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Bring in the whole project (app/, scripts/, models/, etc.)
COPY . .

EXPOSE 5000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
