FROM python:3.13-slim
# build-essential: insightface ships as sdist with a C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# deps first, in their own layer, so a code change doesn't recompile insightface (keep in sync with pyproject)
RUN pip install --no-cache-dir a2wsgi fastapi imagehash jinja2 pillow pillow-heif python-multipart uvicorn wsgidav \
    insightface onnxruntime opencv-python-headless
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
ENV NDRIVE_HOME=/storage
EXPOSE 8484
CMD ["python", "-m", "ndrive", "serve", "--host", "0.0.0.0", "--port", "8484"]
