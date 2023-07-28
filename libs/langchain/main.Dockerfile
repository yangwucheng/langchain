# Use the Python base image
FROM python:3.11.2-bullseye AS builder

ARG POETRY_HOME=/opt/poetry
ARG POETRY_VERSION=1.4.2

# Create a Python virtual environment for Poetry and install it
RUN python3 -m venv ${POETRY_HOME} && \
    $POETRY_HOME/bin/pip install --upgrade pip && \
    $POETRY_HOME/bin/pip install poetry==${POETRY_VERSION}

# Set the working directory for the app
WORKDIR /app

# Copy only the dependency files for installation
COPY pyproject.toml poetry.lock poetry.toml config.py main.py ./

RUN $POETRY_HOME/bin/poetry install
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple --upgrade pymupdf
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple flask==2.2.5
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple python-dotenv
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple unstructured
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple pyyaml==6.0
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple pydantic==1.10.9
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple numpy==1.24.3
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple langsmith==0.0.5
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple aiohttp==3.8.4
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple tenacity==8.2.2
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple openapi_schema_pydantic==1.2.4
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple sqlalchemy==2.0.16
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple numexpr==2.8.4
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple mypy_extensions==1.0.0
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple openai==0.27.8
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple lxml==4.9.2
RUN $POETRY_HOME/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple peewee==3.16.2

RUN mkdir ./source-docs && mkdir ./destination-docs

#CMD ["/bin/bash","-c","cd /app && /opt/poetry/bin/python main.py"]
CMD ["/bin/bash","-c","cd /app && tail -f /dev/null"]