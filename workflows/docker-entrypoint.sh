#!/bin/bash
# Docker entrypoint for CV-SAR SR Pipeline

set -e

echo "🚀 CV-SAR SR Pipeline Starting..."

# GPU 확인
if nvidia-smi &> /dev/null; then
    echo "✅ GPU 감지됨:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    export USE_GPU_FFT=True
else
    echo "⚠️ GPU 없음 - CPU 모드로 실행"
    export USE_GPU_FFT=False
fi

# 메모리 확인
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "💾 시스템 메모리: ${TOTAL_MEM}GB"

# 동적 워커 수 설정
if [ $TOTAL_MEM -ge 32 ]; then
    export MAX_WORKERS=8
elif [ $TOTAL_MEM -ge 16 ]; then
    export MAX_WORKERS=4
else
    export MAX_WORKERS=2
fi
echo "🔧 MAX_WORKERS 설정: ${MAX_WORKERS}"

# S3 설정 확인
if [ -n "$S3_BUCKET" ]; then
    echo "☁️ AWS S3 설정 감지됨"
    
    # AWS CLI 설치 (필요시)
    if ! command -v aws &> /dev/null; then
        echo "AWS CLI 설치 중..."
        apt-get update && apt-get install -y awscli
    fi
    
    # S3에서 입력 데이터 다운로드
    if [ -n "$S3_INPUT_PREFIX" ]; then
        echo "📥 S3에서 입력 데이터 다운로드: s3://$S3_BUCKET/$S3_INPUT_PREFIX"
        aws s3 sync "s3://$S3_BUCKET/$S3_INPUT_PREFIX" /app/data/processed_1/ \
            --exclude "*" --include "*.dim" --include "*.data/*"
    fi
fi

# 입력 파일 확인
DIM_COUNT=$(find /app/data/processed_1 -name "*.dim" 2>/dev/null | wc -l)
echo "📁 발견된 .dim 파일: ${DIM_COUNT}개"

# 환경 변수로 경로 설정
export IN_DIR=/app/data/processed_1
export OUT_DIR=/app/output

# 실행 인수 처리
if [ "$1" = "bash" ]; then
    exec "$@"
elif [ "$1" = "test" ]; then
    echo "🧪 테스트 모드 실행 (1개 파일만)"
    export MAX_FILES=1
    python3 workflows/patch_extractor_production_final.py
elif [ "$1" = "gpu-test" ]; then
    echo "🧪 GPU 테스트 모드"
    export MAX_FILES=1
    python3 workflows/patch_extractor_gpu_enhanced.py
else
    # 기본 실행
    echo "🏃 프로덕션 실행 시작"
    
    # 실행
    python3 "$@" || EXIT_CODE=$?
    
    # S3 업로드 (성공시)
    if [ -z "$EXIT_CODE" ] && [ -n "$S3_BUCKET" ] && [ -n "$S3_OUTPUT_PREFIX" ]; then
        echo "📤 S3로 결과 업로드: s3://$S3_BUCKET/$S3_OUTPUT_PREFIX"
        aws s3 sync /app/output/ "s3://$S3_BUCKET/$S3_OUTPUT_PREFIX" \
            --exclude "*.log"
        
        # 업로드 완료 후 로컬 파일 정리 (옵션)
        if [ "$CLEANUP_AFTER_UPLOAD" = "true" ]; then
            echo "🧹 로컬 출력 정리"
            find /app/output -name "*.npy" -delete
        fi
    fi
    
    exit ${EXIT_CODE:-0}
fi 