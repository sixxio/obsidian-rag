FROM python:3.11-slim-buster

COPY builds/init/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY /app app
COPY /data /data

CMD ["python", "app/init_db.py"]
