FROM python:3.11-slim-buster

COPY builds/bot/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY /bot/main.py /app/main.py

CMD ["python", "/app/main.py"]