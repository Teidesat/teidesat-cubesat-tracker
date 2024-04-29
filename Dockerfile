FROM ubuntu:jammy


# Install system dependencies

RUN apt update  \
    && apt install --assume-yes \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libqt5gui5 \
        libusb-1.0-0-dev \
        python3 \
        python3-pip \
        unzip \
        wget \
    && apt clean


# Install Python dependencies

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt


# Install ZWO ASI SDK

WORKDIR /tmp
RUN wget --output-document=ASI_Camera_SDK.zip \
        --output-file=/dev/null \
        https://e.zwoastro.com/attachments/1711105859128-ILxbipvA78.zip\
    && unzip /tmp/ASI_Camera_SDK.zip \
    && tar --extract --verbose --bzip2 --file ASI_Camera_SDK/ASI_linux_mac_SDK_V*.tar.bz2 \
    && rm --recursive ASI_Camera_SDK.zip ASI_Camera_SDK \
    && install ASI_linux_mac_SDK_V*/lib/asi.rules /lib/udev/rules.d \
    && cp ASI_linux_mac_SDK_V*/lib/x64/* /usr/lib64/

ENV ZWO_ASI_LIB=/usr/lib64/libASICamera2.so


## Copy the rest of the application
#
#COPY . /app
#WORKDIR /app
#
#ENTRYPOINT ["python3", "main.py"]
