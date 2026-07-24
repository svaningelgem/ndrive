FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV NDRIVE_HOME=/storage
EXPOSE 8484
CMD ["python", "-m", "ndrive", "serve", "--host", "0.0.0.0", "--port", "8484"]
