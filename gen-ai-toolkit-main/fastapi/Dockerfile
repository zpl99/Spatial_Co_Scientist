FROM python:3.11-slim

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install necessary tools, zsh, and set up locale
RUN apt-get update && \
    apt-get install --no-install-recommends -y zsh wget git curl locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    # Cleanup apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /gait
COPY pyproject.toml /gait/pyproject.toml
COPY gait /gait/gait
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip
RUN pip install ".[fastapi]"
COPY fastapi /gait/fastapi
COPY fastapi/.env /gait/fastapi/.env
EXPOSE 8000
WORKDIR /gait/fastapi
ENV OLLAMA_HOST=http://host.docker.internal:11434
CMD ["fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0", "--app", "app"]