#!/bin/bash
# Usage:
#   bash docker-run.sh              # interactive shell
#   bash docker-run.sh phase0       # run phase 0 baselines
#   bash docker-run.sh phase2 2.4   # run phase 2 experiment 2.4
#   bash docker-run.sh phase3 5     # run phase 3 inference (5 days)

IMAGE_NAME="btc-regime"
CONTAINER_NAME="btc-regime-run"

# Build if image doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building image (first time, ~10 min)..."
    docker build -t $IMAGE_NAME .
fi

# Remove old container if exists
docker rm -f $CONTAINER_NAME 2>/dev/null

CMD="bash"
case "$1" in
    phase0) CMD="python3 experiments/phase0_baselines.py" ;;
    phase1) CMD="python3 experiments/phase1_regime_classifier.py" ;;
    phase2) CMD="python3 experiments/phase2_finetune.py --exp ${2:-2.4}" ;;
    phase3) CMD="python3 experiments/phase3_inference.py --days ${2:-5}" ;;
    all)    CMD="python3 experiments/phase0_baselines.py && python3 experiments/phase2_finetune.py --exp 2.4 && python3 experiments/phase3_inference.py --days 5" ;;
esac

docker run -it --gpus all \
    --name $CONTAINER_NAME \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/btc-regime/data \
    -v $(pwd)/models:/workspace/btc-regime/models \
    -v $(pwd)/results:/workspace/btc-regime/results \
    -v $(pwd)/experiments:/workspace/btc-regime/experiments \
    -v $(pwd)/pipeline:/workspace/btc-regime/pipeline \
    -v $(pwd)/run_pipeline.py:/workspace/btc-regime/run_pipeline.py \
    $IMAGE_NAME \
    bash -c "$CMD"
