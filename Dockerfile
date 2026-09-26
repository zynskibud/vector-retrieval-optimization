# Benchmark environment: Ubuntu 24.04 on arm64 with the four toolchains.
# The repository is bind-mounted at /work at run time (see docker-compose.yml);
# this image holds only tools, no project code.
FROM ubuntu:24.04

ARG GO_VERSION=1.27.1
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/go/bin:/usr/local/cargo/bin:/root/.local/bin:$PATH \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_ADDOPTS="-p no:cacheprovider"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang cmake git curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*

# uv (Python project manager) and a pinned CPython 3.12
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/
RUN uv python install 3.12

# Go
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-arm64.tar.gz" | tar -C /usr/local -xz

# Rust (stable), shared install so every user of the image sees it
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path \
    && rustup component add clippy rustfmt \
    && rustup --version && cargo --version && cargo clippy --version

WORKDIR /work
CMD ["bash"]
