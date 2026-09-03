FROM python:3.11-slim

WORKDIR /app

COPY eval_requirements.txt .
RUN pip install --no-cache-dir -r eval_requirements.txt

COPY eval_server.py .
COPY eval_validation.py .
COPY start.sh .
RUN chmod +x start.sh

RUN mkdir -p /data/audio_fma

EXPOSE 5000

CMD ["./start.sh"]
