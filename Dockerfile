FROM python:3.9.6

# Install Git LFS
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install \
    git-lfs pull

WORKDIR /app

COPY requirements.txt .

SHELL ["/bin/bash", "-c"]
RUN python3 -m venv .venv
RUN source .venv/bin/activate
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3002

ENTRYPOINT [ "python3", "./api/index.py" ]