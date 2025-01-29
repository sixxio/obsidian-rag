FROM python:3.11-slim-buster

WORKDIR /usr/src/app

RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary

CMD mlflow server \
  --backend-store-uri ${BACKEND_URI} \
  --host 0.0.0.0