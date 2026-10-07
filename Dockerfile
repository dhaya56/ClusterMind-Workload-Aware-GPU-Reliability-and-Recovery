FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

RUN groupadd --system clustermind \
    && useradd --system --gid clustermind --create-home clustermind

COPY requirements.txt ${APP_HOME}/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --requirement ${APP_HOME}/requirements.txt

COPY app ${APP_HOME}/app
COPY simulator ${APP_HOME}/simulator
COPY configs ${APP_HOME}/configs
COPY runbooks ${APP_HOME}/runbooks
COPY scripts ${APP_HOME}/scripts
COPY data ${APP_HOME}/data
COPY PROJECT_BLUEPRINT.md DATASET.md ${APP_HOME}/

RUN mkdir -p \
        ${APP_HOME}/data/generated \
        ${APP_HOME}/data/artifacts \
        ${APP_HOME}/app/ml/artifacts \
    && chown -R clustermind:clustermind ${APP_HOME}

USER clustermind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3); raise SystemExit(0 if response.status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]