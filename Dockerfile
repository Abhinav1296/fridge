# Zero-Waste Kitchen Agent — production image.
#
# Runs the Flask app under gunicorn. State lives in SQLite by default (mount a
# volume at /data), or point TURSO_DATABASE_URL / TURSO_AUTH_TOKEN at a hosted
# libSQL and no volume is needed. All config is read from the environment at
# call time — never bake secrets into this image.

FROM python:3.12-slim

# - Don't write .pyc files; stream stdout/stderr straight to the container log.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    DATABASE_PATH=/data/smartfridge.db

WORKDIR /app

# Install dependencies first so layer caching survives source-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

# Persistent SQLite lives here; run as an unprivileged user that owns it.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

VOLUME ["/data"]
EXPOSE 5000

# Liveness/readiness — the app exposes /health (200 healthy, 503 if the DB is down).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.environ.get('PORT','5000'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"

# ONE gthread worker with a pool of threads. Socket.IO runs in threading mode and
# keeps its connected-client set in process memory, so a broadcast from a second
# worker would never reach clients attached to the first. Stay single-worker until
# real-time is fan-out across a Redis message queue (SocketIO(message_queue=...)),
# then it becomes safe to raise --workers. Threads still give I/O-bound concurrency.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --worker-class gthread --timeout 120 app:app"]
