#!/usr/bin/env bash

cat <<EOT > job.sh
#!/bin/bash

#SBATCH --job-name=twist_train
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --time=16:00:00
#SBATCH --output=pace_logs/twist_train_%j.log

source ~/.bashrc
bash "run_singularity.sh" "$1" "${@:2}"
EOT

cat < job.sh
sbatch < job.sh
rm job.sh
