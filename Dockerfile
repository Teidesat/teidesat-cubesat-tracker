FROM ubuntu:jammy

RUN apt update  \
    && apt install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libqt5gui5 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements*

COPY . /app
WORKDIR /app

ENTRYPOINT ["python3", "main.py"]
