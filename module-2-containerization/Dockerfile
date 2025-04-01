# ======= STAGE 1: Builder =======

ARG IMAGE=python:3.9-slim@sha256:e52ca5f579cc58fed41efcbb55a0ed5dccf6c7a156cba76acfb4ab42fc19dd00
ARG USERNAME=mluser
ARG USER_UID=1111
ARG USER_GID=1111

FROM ${IMAGE} AS builder

ARG USERNAME
ARG USER_UID
ARG USER_GID


WORKDIR /app

# Upgrade pip and install packages 

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip==25.0.1 && \
    pip install --no-cache-dir -r requirements.txt
        

# ======= STAGE 2: Final runtime =======

FROM ${IMAGE} AS runtime

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Creates non-root user

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}


# Copy installed packages and Jupyter schema files from builder stage

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/jupyter /usr/local/share/jupyter

                
                
WORKDIR /app

COPY notebooks/ /app/notebooks/

RUN chown -R ${USERNAME}:${USERNAME} /app

USER $USERNAME

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password="]

LABEL maintainer="https://hub.docker.com/u/mtech001" \
      version="1.0" \
      description="MLE environment with TensorFlow and Jupyter"
