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
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$IMAGE_NAME.tar ]"; then
        echo "[Error] The '$IMAGE_NAME' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

#==
# Main
#==
IMAGE_NAME="twist"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/.env.cluster"
check_singularity_image_exists

# Setup directories
setup_directories
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# Remove any existing extracted image to ensure clean extraction
rm -rf $TMPDIR/$IMAGE_NAME

# Extract fresh image from tar
tar -xf $CLUSTER_SIF_PATH/$IMAGE_NAME.tar -C $TMPDIR
echo "Singularity image extracted to $TMPDIR/$IMAGE_NAME"

# Get args
PROJECT_DIR="$1"
EXTRA_ARGS="${@:2}"

echo "Configuration:"
echo "  Project directory: $PROJECT_DIR"
echo "  SIF directory: $CLUSTER_SIF_PATH"
echo "  Dataset directory: $CLUSTER_DATASET_DIR"
echo "  Python executable: $CLUSTER_PYTHON_EXECUTABLE"
echo "  Extra arguments: $EXTRA_ARGS"
echo ""

# Check if Isaac Gym is available
echo "Starting training in singularity container..."

# Execute the training script in the singularity container
singularity exec \
    --nv \
    --writable-tmpfs \
    -B $PROJECT_DIR:/workspace/TWIST:rw \
    -B $CLUSTER_DATASET_DIR:/workspace/TWIST_Dataset:ro \
    -B $CLUSTER_GCC_TOOLCHAIN_DIR/bin/gcc:/usr/bin/gcc:ro \
    -B $CLUSTER_GCC_TOOLCHAIN_DIR/bin/g++:/usr/bin/g++:ro \
    -B $CLUSTER_GCC_TOOLCHAIN_DIR/lib/gcc:/usr/lib/gcc:ro \
    -B $CLUSTER_GCC_TOOLCHAIN_DIR/include:/usr/include:ro \
    -B $CLUSTER_GCC_TOOLCHAIN_DIR/lib64:/host-gcc-lib64:ro \
    $TMPDIR/$IMAGE_NAME \
    bash -c "
        eval \"\$(conda shell.bash hook)\"
        conda activate twist
        export LD_LIBRARY_PATH=/host-gcc-lib64:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
        export LIBRARY_PATH=/host-gcc-lib64:\$LIBRARY_PATH
        export CPATH=/usr/lib/gcc/x86_64-linux-gnu/11/include:\$CPATH
        export CC=/usr/bin/gcc
        export CXX=/usr/bin/g++
        python --version
        export PYTHONPATH=/workspace/isaacgym/python:/workspace/TWIST_Original/rsl_rl:/workspace/TWIST_Original/pose:/workspace/TWIST/legged_gym:\$PYTHONPATH
        cd /workspace/TWIST
        cd rsl_rl && pip install -e . && cd ..
        cd legged_gym && pip install -e . --no-deps
        cd ..
        export WANB_API_KEY=$CLUSTER_WANDB_API_KEY
        python $CLUSTER_PYTHON_EXECUTABLE $EXTRA_ARGS
    "



