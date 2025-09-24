# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model (after installing spacy)
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY app ./app

# Expose port (inside container)
EXPOSE 5000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
