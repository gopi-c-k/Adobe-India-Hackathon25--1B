FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch and other dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-download MPNet model (cached offline)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
ENV TRANSFORMERS_OFFLINE=1

COPY app.py .

# Create mount points
RUN mkdir -p /app/input /app/output

ENV PERSONA="Default Persona"
ENV JOB="Default Job"

CMD ["python", "app.py"]
