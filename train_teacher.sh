
# bash train_teacher.sh 0927_twist_teacher cuda:0

# export LD_LIBRARY_PATH=/home/keseterg/Expansion1/miniforge3/envs/twist/lib:$LD_LIBRARY_PATH
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
                --entity RoboRambler \
                # --debug
                # --resume \
                # --resumeid xxx
