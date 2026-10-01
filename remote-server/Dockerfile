FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/app/models/parakeet-unified-en-0.6b-Q8_0.gguf \
    TRANSCRIBE_LIBRARY=/opt/transcribe/lib/libtranscribe.so \
    PYTHONPATH=/opt/transcribe.cpp/bindings/python/src \
    PORT=10000

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/transcribe.cpp
RUN git init \
    && git remote add origin https://github.com/handy-computer/transcribe.cpp.git \
    && git fetch --depth 1 origin 4807edaf210d0d7e8a6f7fb2a44b65966a2797f0 \
    && git checkout --detach FETCH_HEAD \
    && cmake -S . -B /opt/transcribe-build -DCMAKE_BUILD_TYPE=Release \
      -DTRANSCRIBE_BUILD_SHARED=ON \
      -DTRANSCRIBE_BUILD_TESTS=OFF \
      -DTRANSCRIBE_BUILD_EXAMPLES=OFF \
      -DTRANSCRIBE_BUILD_TOOLS=OFF \
      -DTRANSCRIBE_USE_OPENMP=OFF \
      -DTRANSCRIBE_USE_SYSTEM_BLAS=OFF \
      -DTRANSCRIBE_VULKAN=OFF \
      -DTRANSCRIBE_CUDA=OFF \
      -DTRANSCRIBE_HIP=OFF \
      -DTRANSCRIBE_GGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DTRANSCRIBE_X86_CONSERVATIVE=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/transcribe \
      -DCMAKE_INSTALL_LIBDIR=lib \
    && cmake --build /opt/transcribe-build --target transcribe -j2 \
    && cmake --install /opt/transcribe-build

RUN python -m pip install --no-cache-dir \
      'fastapi==0.115.12' 'uvicorn[standard]==0.34.0' 'httpx==0.28.1' \
    && python -c 'import transcribe_cpp; assert transcribe_cpp.backend_available("cpu"), transcribe_cpp.backends()' \
    && mkdir -p /app/models \
    && curl -fL --retry 3 \
      'https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q8_0.gguf' \
      -o /app/models/parakeet-unified-en-0.6b-Q8_0.gguf

WORKDIR /app
COPY server.py model_worker.py test_server.py ./
RUN API_TOKEN=test-secret python -c 'import transcribe_cpp; assert transcribe_cpp.backend_available("cpu"), transcribe_cpp.backends()' \
    && python -m unittest -v test_server

EXPOSE 10000
CMD ["python", "server.py"]
