FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock.txt requirements-service.txt requirements-paper.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt
COPY chronic ./chronic
COPY service ./service
COPY paper_models ./paper_models
COPY configs ./configs
COPY artifacts/v2/survey.joblib artifacts/v2/glucose.joblib ./artifacts/v2/
COPY artifacts/table_heart_v1/table_model.joblib ./artifacts/table_heart_v1/
COPY artifacts/papers ./artifacts/papers
RUN useradd --uid 10001 --create-home chronic && mkdir /data && chown chronic:chronic /data
ENV CHRONIC_STORAGE=/data CHRONIC_HOST=0.0.0.0 CHRONIC_PORT=8765
USER chronic
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz',timeout=3)"
CMD ["python", "-m", "service.server"]
