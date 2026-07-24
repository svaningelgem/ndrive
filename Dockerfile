FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libheif-examples && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV NDRIVE_HOME=/storage
EXPOSE 8484
CMD ["python", "-m", "ndrive", "serve", "--host", "0.0.0.0", "--port", "8484"]
