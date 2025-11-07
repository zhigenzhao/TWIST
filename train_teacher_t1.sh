#!/bin/bash
# T1 Teacher Policy Training Script
# Usage: bash train_teacher_t1.sh t1_teacher_exp1 cuda:0

# Set library path for Isaac Gym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd legged_gym/legged_gym/scripts

exptid=$1
device=$2

task_name="t1_priv_mimic"
proj_name="t1_priv_mimic"

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                # --resume \
                # --debug
                # --resumeid xxx
