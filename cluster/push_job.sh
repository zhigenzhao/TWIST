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

echo -e "${GREEN}=== PACE Training Job Submission ===${NC}"
echo ""
echo "Configuration:"
echo "  Local directory: $LOCAL_DIR"
echo "  Remote: $REMOTE_FULL:$REMOTE_DIR"
echo ""

# Step 1: Rsync local code to remote
echo -e "${YELLOW}[1/2] Syncing local code to remote server...${NC}"

rsync -avzh --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='runs/' \
    --exclude='logs/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*.log' \
    --exclude='wandb/' \
    --exclude='wandb_download/' \
    --exclude='*.egg-info/' \
    --filter=':- .gitignore' \
    "$LOCAL_DIR/" "$REMOTE_FULL:$REMOTE_DIR/"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Rsync failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Rsync completed successfully!${NC}"
echo ""

# Step 2: Submit SLURM job
echo -e "${YELLOW}[2/2] Submitting SLURM job...${NC}"

echo -e "${GREEN}=== Job submission complete! ===${NC}"
