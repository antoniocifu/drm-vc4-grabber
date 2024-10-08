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
RUN git clone https://github.com/antoniocifu/drm-vc4-grabber.git .

# Install target for compilation
RUN rustup target add x86_64-unknown-linux-gnu

RUN mkdir /build
RUN cargo build --release --target x86_64-unknown-linux-gnu

#RUN cp /drm-vc4-grabbertarget/x86_64-unknown-linux-gnu/release/drm-vc4-grabber /build/drm-vc4-grabber

# Default command to work inside docker container (usseful in LibreElec)
CMD ["/bin/bash"]