# 也可以用 Docker 跑：
#   docker build -t ev-doctor -f deploy/Dockerfile .
#   docker run -d --name ev-doctor -p 8000:8000 -v evdoctor-data:/app/data -v $PWD/config.json:/app/config.json ev-doctor
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN rm -rf /app/data/evdoctor.db /app/data/backups /app/data/server.log /app/__pycache__ \
 && useradd -r -u 10001 evdoctor && mkdir -p /app/data && chown -R evdoctor /app/data
USER evdoctor
ENV EVDOC_HOST=0.0.0.0 EVDOC_PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["python", "server.py"]
