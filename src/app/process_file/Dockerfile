FROM python:3.12-slim-bullseye

RUN apt-get update \
    && pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install --no-cache-dir --requirement /app/requirements.txt

COPY . /app

EXPOSE 5010

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5010", "--reload"]