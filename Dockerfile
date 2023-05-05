FROM python:3.9.6

WORKDIR /app

COPY requirements.txt .

SHELL ["/bin/bash", "-c"]
RUN python3 -m venv .venv
RUN source .venv/bin/activate
RUN pip install -r requirements.txt

EXPOSE 5000

COPY . .

ENTRYPOINT [ "python3", "./api/index.py" ]