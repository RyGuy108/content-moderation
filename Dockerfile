FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheel
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY scripts/ scripts/
RUN bash scripts/fetch_data.sh
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "modpipe.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
