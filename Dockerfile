FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

CMD exec uvicorn gtn_copliot.server:app --host 0.0.0.0 --port $PORT

