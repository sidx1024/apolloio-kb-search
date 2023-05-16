FROM python:3.9.6

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 3002

ENTRYPOINT [ "python3", "./api/index.py" ]
