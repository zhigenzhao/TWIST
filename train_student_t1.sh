#!/bin/bash
# T1 Student Policy Training Script
# Usage: bash train_student_t1.sh t1_student_exp1 t1_teacher_exp1 cuda:0

# Set library path for Isaac Gym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd legged_gym/legged_gym/scripts

exptid=$1
teacher_exptid=$2
device=$3

task_name="t1_stu_rl"
proj_name="t1_stu_rl"

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_id "${teacher_exptid}" \
                # --resume \
                # --debug
                # --resumeid xxx
