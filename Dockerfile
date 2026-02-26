FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY multijet_tsne.py .

ENTRYPOINT ["python", "/app/multijet_tsne.py"]
