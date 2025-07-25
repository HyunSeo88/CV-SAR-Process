# Dockerfile for CV-SAR SR Patch Extraction Pipeline
# AWS EC2 Deployment with GPU Support

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    git \
    python3.10 \
    python3-pip \
    openjdk-11-jdk \
    libgfortran5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install SNAP (ESA Sentinel Application Platform)
ENV SNAP_VERSION=9.0.0
ENV SNAP_INSTALLER=esa-snap_sentinel_linux-${SNAP_VERSION}.sh
ENV SNAP_HOME=/opt/snap

RUN wget -q https://download.esa.int/step/snap/${SNAP_VERSION}/${SNAP_INSTALLER} && \
    chmod +x ${SNAP_INSTALLER} && \
    ./${SNAP_INSTALLER} -q -dir ${SNAP_HOME} && \
    rm ${SNAP_INSTALLER}

# Configure SNAP for headless operation
RUN ${SNAP_HOME}/bin/snap --nosplash --nogui --modules --update-all 2>&1 | true

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

# Install CuPy for GPU acceleration
RUN pip3 install --no-cache-dir cupy-cuda11x

# Configure snappy Python bindings
ENV PYTHONPATH="${SNAP_HOME}/snap/modules/org-esa-snap-snap-python.jar:${PYTHONPATH}"
RUN cd ${SNAP_HOME}/bin && \
    ./snappy-conf /usr/bin/python3 && \
    cd /root/.snap/snap-python/snappy && \
    python3 setup.py install

# Copy application code
COPY workflows/ /app/workflows/
COPY data/ /app/data/

# Create output directory
RUN mkdir -p /app/output

# Set environment variables for production
ENV MAX_FILES=None
ENV USE_GPU_FFT=True
ENV MAX_WORKERS=4
ENV MIN_CROSS_POL_COHERENCE=0.95
ENV MIN_ENERGY_RATIO=0.95

# AWS S3 configuration (set at runtime)
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV S3_BUCKET=""
ENV S3_INPUT_PREFIX=""
ENV S3_OUTPUT_PREFIX=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import esa_snappy; print('OK')" || exit 1

# Entry point script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python3", "workflows/patch_extractor_production_final.py"] 