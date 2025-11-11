#!/bin/bash
# Usage:
# bash play_xrt.sh <mode> [xrt_file] [options]
#
# Examples:
#   bash play_xrt.sh realtime                    # Real-time XRT streaming
#   bash play_xrt.sh file /path/to/motion.pkl    # Replay from file
#   bash play_xrt.sh realtime --record           # With video recording
#   bash play_xrt.sh realtime --cpu              # Run on CPU (for RTX 5090 compatibility)

# Fix library path for Isaac Gym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Configuration
proj_name="t1_student"
exptid="1"
checkpoint=10000
task_name="t1_stu_rl"

# Parse arguments
mode=${1:-realtime}  # Default to realtime if not specified
xrt_file=${2}

cd legged_gym/legged_gym/scripts

# Build the command
cmd="python play_xrt.py \
    --task ${task_name} \
    --proj_name ${proj_name} \
    --exptid ${exptid} \
    --checkpoint ${checkpoint} \
    --mode ${mode}"

# Add xrt_file if provided and mode is file
if [ "$mode" = "file" ] && [ -n "$xrt_file" ]; then
    cmd="$cmd --xrt_file ${xrt_file}"
fi

# Check for flags in all arguments (starting from position 2 if mode is realtime, 3 if file)
start_pos=2
if [ "$mode" = "file" ]; then
    start_pos=3
fi

for arg in "${@:$start_pos}"; do
    if [ "$arg" = "--record" ] || [ "$arg" = "-r" ]; then
        cmd="$cmd --record_video --record_log"
    elif [ "$arg" = "--headless" ] || [ "$arg" = "-h" ]; then
        cmd="$cmd --headless"
    elif [ "$arg" = "--use_jit" ] || [ "$arg" = "-j" ]; then
        cmd="$cmd --use_jit"
    elif [ "$arg" = "--cpu" ] || [ "$arg" = "-c" ]; then
        cmd="$cmd --sim_device cpu --rl_device cpu"
        echo "Running in CPU mode (for RTX 5090 compatibility)"
    fi
done

# Print the command for debugging
echo "Running: $cmd"
echo ""

# Run the command
eval $cmd
