#!/bin/bash
# Usage:
#   bash docker-run.sh              # interactive shell
#   bash docker-run.sh eval         # full TimesFM evaluation (core research)
#   bash docker-run.sh phase05      # Phase 0.5 smoke test (decision gate)
#   bash docker-run.sh phase2       # Phase 2 progressive fine-tune
#   bash docker-run.sh phase3       # Phase 3 risk signals
#   bash docker-run.sh all          # eval + phase05 + phase2 + phase3

IMAGE_NAME="crypto-regime"
CONTAINER_NAME="crypto-regime-run"

# Build if image doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building image (first time, ~10 min)..."
    docker build -t $IMAGE_NAME .
fi

# Remove old container if exists
docker rm -f $CONTAINER_NAME 2>/dev/null

CMD="bash"
case "$1" in
    eval)     CMD="python3 experiments/eval_timesfm.py" ;;
    phase05)  CMD="python3 experiments/phase05_smoke_test.py" ;;
    phase2)   CMD="python3 experiments/phase2_finetune.py --stage ${2:-progressive}" ;;
    phase3)   CMD="python3 experiments/phase3_risk_signals.py --model ${2:-auto}" ;;
    baseline) CMD="python3 experiments/phase0_baselines.py" ;;
    all)      CMD="python3 experiments/eval_timesfm.py && \
                   python3 experiments/phase05_smoke_test.py && \
                   echo '=== Check phase05 results before continuing ===' && \
                   python3 experiments/phase3_risk_signals.py --model zero-shot" ;;
esac

echo "Running: $CMD"
echo "---"

docker run -it --gpus all \
    --name $CONTAINER_NAME \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/crypto-regime/data \
    -v $(pwd)/models:/workspace/crypto-regime/models \
    -v $(pwd)/results:/workspace/crypto-regime/results \
    -v $(pwd)/experiments:/workspace/crypto-regime/experiments \
    -v $(pwd)/pipeline:/workspace/crypto-regime/pipeline \
    -v $(pwd)/run_pipeline.py:/workspace/crypto-regime/run_pipeline.py \
    -v $(pwd)/reports:/workspace/crypto-regime/reports \
    -v $(pwd)/requirements.txt:/workspace/crypto-regime/requirements.txt \
    $IMAGE_NAME \
    bash -c "$CMD"
