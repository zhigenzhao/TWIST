#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load configuration from .env.cluster
ENV_FILE="$SCRIPT_DIR/.env.cluster"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Configuration file not found: $ENV_FILE"
    echo "Please create .env.cluster appropriately."
    exit 1
fi

# Source the environment file
source "$ENV_FILE"

REMOTE_FULL="$CLUSTER_LOGIN"
REMOTE_DIR="$CLUSTER_BASE_DIR"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Pulling Latest Model from Remote Server ===${NC}"
echo ""
echo "Configuration:"
echo "  Remote: $REMOTE_FULL:$REMOTE_DIR/legged_gym/logs"
echo ""

# Find the latest model (highest checkpoint number) for each run
echo -e "${YELLOW}Finding latest model for each run on remote server...${NC}"

# Get list of latest models (one per run directory)
LATEST_MODELS=$(ssh "$REMOTE_FULL" "
cd $REMOTE_DIR/legged_gym/logs || exit 1
# Find all run directories (2 levels deep: proj/run/)
for run_dir in \$(find . -mindepth 2 -maxdepth 2 -type d); do
    # Find the model with the largest checkpoint number in this run
    latest=\$(ls \$run_dir/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n \"\$latest\" ]; then
        echo \"\$latest\"
    fi
done
")

if [ -z "$LATEST_MODELS" ]; then
    echo -e "${RED}Error: No model files found on remote server!${NC}"
    exit 1
fi

# Count number of models to download
MODEL_COUNT=$(echo "$LATEST_MODELS" | wc -l)
echo "Found $MODEL_COUNT run(s) with models to download"
echo ""

# Download each model
COUNTER=1
echo "$LATEST_MODELS" | while read -r RELATIVE_MODEL; do
    # Extract project and run name
    PROJ_NAME=$(echo "$RELATIVE_MODEL" | cut -d'/' -f2)
    RUN_NAME=$(echo "$RELATIVE_MODEL" | cut -d'/' -f3)
    MODEL_FILE=$(basename "$RELATIVE_MODEL")

    echo -e "${YELLOW}[$COUNTER/$MODEL_COUNT] Syncing model...${NC}"
    echo "  Project: $PROJ_NAME"
    echo "  Run: $RUN_NAME"
    echo "  Model: $MODEL_FILE"

    # Construct paths
    REMOTE_MODEL="$REMOTE_DIR/legged_gym/logs/$RELATIVE_MODEL"
    LOCAL_MODEL_PATH="$LOCAL_DIR/legged_gym/logs/$RELATIVE_MODEL"

    # Create local directory structure
    mkdir -p "$(dirname "$LOCAL_MODEL_PATH")"

    # Download the model
    rsync -azh --progress \
        "$REMOTE_FULL:$REMOTE_MODEL" "$LOCAL_MODEL_PATH"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Rsync failed for $MODEL_FILE!${NC}"
    else
        echo -e "${GREEN}Downloaded successfully!${NC}"
    fi
    echo ""

    COUNTER=$((COUNTER + 1))
done

echo -e "${GREEN}All models pulled successfully!${NC}"
echo ""
