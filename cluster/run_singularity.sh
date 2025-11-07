#!/usr/bin/env bash

echo "====================================="
echo "TWIST Singularity Execution Script"
echo "====================================="
echo "Called from directory: $1"
echo "Arguments: ${@:2}"
echo ""

setup_directories() {
    # Check and create directories for Isaac Sim cache
    echo "Setting up cache directories..."
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "  Created directory: $dir"
        fi
    done
    echo "Cache directories ready."
    echo ""
}

check_singularity_image_exists() {
    image_name="twist_gym"
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

#==
# Main
#==
IMAGE_NAME="twist_gym"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/.env.cluster"
check_singularity_image_exists

# Setup directories
setup_directories
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR
tar -xf $CLUSTER_SIF_PATH/$IMAGE_NAME.tar  -C $TMPDIR
echo "Singularity image extracted to $TMPDIR/$IMAGE_NAME"

# Get args
PROJECT_DIR="$1"
EXTRA_ARGS="${@:2}"

echo "Configuration:"
echo "  Project directory: $PROJECT_DIR"
echo "  SIF directory: $CLUSTER_SIF_DIR"
echo "  Dataset directory: $CLUSTER_DATASET_DIR"
echo "  Python executable: $CLUSTER_PYTHON_EXECUTABLE"
echo "  Extra arguments: $EXTRA_ARGS"
echo ""

# Check if Isaac Gym is available
echo "Starting training in singularity container..."

# Execute the training script in the singularity container
singularity exec \
    --nv \
    --containall \
    -B $PROJECT_DIR:/workspace/TWIST:rw \
    -B $CLUSTER_DATASET_DIR:/workspace/TWIST_Dataset:ro \
    $TMPDIR/$IMAGE_NAME \
    bash -c "cd /workspace/TWIST && sh install.sh && python $CLUSTER_PYTHON_EXECUTABLE $EXTRA_ARGS"



