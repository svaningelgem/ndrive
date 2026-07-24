FROM python:3.13-slim
# build-essential: insightface ships as sdist with a C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[faces]"
ENV NDRIVE_HOME=/storage
EXPOSE 8484
CMD ["python", "-m", "ndrive", "serve", "--host", "0.0.0.0", "--port", "8484"]
