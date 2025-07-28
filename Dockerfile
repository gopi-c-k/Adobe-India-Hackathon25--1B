FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-download ONLY all-mpnet-base-v2 model
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2'); AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')"
ENV TRANSFORMERS_OFFLINE=1

COPY app.py .

RUN mkdir /app/input

ENV PERSONA="Default Persona"
ENV JOB="Default Job"

CMD ["python", "app.py"]
