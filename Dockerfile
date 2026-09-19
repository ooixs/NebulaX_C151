FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY common ./common
COPY Door/code ./Door/code
COPY Door/model ./Door/model
COPY ACV/code ./ACV/code
COPY ACV/model ./ACV/model
COPY ["Rail Corrugation/code", "./Rail Corrugation/code"]
COPY ["Rail Corrugation/model", "./Rail Corrugation/model"]
COPY rail_corrugation ./rail_corrugation
COPY SHM/code ./SHM/code
COPY SHM/model ./SHM/model
COPY predictions ./predictions

RUN mkdir -p /workspace/app/.local && chown 65532:65532 /workspace/app/.local

USER 65532

EXPOSE 8080

CMD ["python", "app/server.py"]
