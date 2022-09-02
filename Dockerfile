FROM python:3.9
LABEL "product classification"="" \
    "maintener"="Barthelemy Pavy" \
    "version"="1.0" \
    "description"=""
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt

# Define env variable for metaflow
ENV USERNAME="user"
# Auth build arguments
ARG USER="user"

# Define UID/GID
ARG UID="1000"
ARG GID="1000"

# Upgrade image
RUN apt-get update && \
    apt-get dist-upgrade --assume-yes && \
    apt-get autoremove --assume-yes --purge

# Setup the user with root privileges.
RUN apt-get install -y sudo && \
    useradd -m -s /bin/bash -N -u $UID $USER && \
    chmod g+w /etc/passwd && \
    echo "${USER}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers

# Install needed packages and clean cache
RUN apt-get install --assume-yes --no-install-recommends \
    make \
    automake \
    gcc \
    curl \
    jq \
    moreutils \
    git \
    python3-dev \
    locales \
    ffmpeg \
    software-properties-common
RUN apt-get clean --assume-yes

# Configure locales
RUN locale-gen en_US.UTF-8

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install poetry
RUN python3 -m pip install poetry

RUN pip install nltk

# Prevent apt-get cache from being persisted to this layer.
RUN rm -rf /var/lib/apt/lists/*

# Make the default shell bash (vs "sh") for a better Jupyter terminal UX.
ENV SHELL=/bin/bash

# Copy the project in the image
COPY . /home/$USER/product_classification


RUN chown 1000 /home/$USER/product_classification && \
    chgrp 1000 /home/$USER/product_classification

USER $UID

WORKDIR /home/$USER/product_classification

RUN ./post_install.sh

ENTRYPOINT ["bash"]
