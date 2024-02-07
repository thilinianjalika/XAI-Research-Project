
# Define function directory
ARG FUNCTION_DIR="/function"

FROM python:3.9-buster as build-image

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Install pip dependencies
COPY cpu-requirements.txt .
RUN pip install -r cpu-requirements.txt --index-url https://download.pytorch.org/whl/cpu --target ${FUNCTION_DIR}
COPY requirements.txt .
RUN pip install -r requirements.txt --target ${FUNCTION_DIR}
# Install the runtime interface client
RUN pip install \
        --target ${FUNCTION_DIR} \
        awslambdaric

# Multi-stage build: grab a fresh copy of the base image
FROM python:3.9-buster

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the build image dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Download nltk data
RUN python3 -m nltk.downloader --dir /usr/share/nltk_data wordnet punkt stopwords averaged_perceptron_tagger tagsets

# copy function code
COPY app/models ${FUNCTION_DIR}/models
COPY app/configs ${FUNCTION_DIR}/configs
COPY app/src ${FUNCTION_DIR}/src
COPY app/handlers ${FUNCTION_DIR}/handlers
COPY app/app.py ${FUNCTION_DIR}/app.py

# Setting the entry point
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler" ]

