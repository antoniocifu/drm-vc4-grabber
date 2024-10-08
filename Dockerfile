FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libdrm-dev \
    libgbm-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust with rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo and rustc to PATH
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /drm-vc4-grabber
RUN git clone https://github.com/rudihorn/drm-vc4-grabber.git .

# Install target for compilation
RUN rustup target add x86_64-unknown-linux-gnu

RUN cargo build --release --target x86_64-unknown-linux-gnu

# Default command to work inside docker container (usseful in LibreElec)
CMD ["/bin/bash"]