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

# 2 workers × 4 threads is a sane default for a small I/O-bound app; override at deploy.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 app:app"]
