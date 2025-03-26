#!/bin/bash
#SBATCH --job-name=ZeroTIG_DID
#SBATCH --account=coms030646
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100
#SBATCH --mem=20GB
#SBATCH --time=7-00:00:00
#SBATCH --output=log-job_test.out
#SBATCH --error=log-err_test.err

. ~/initMamba.sh
conda activate py39
cd /user/home/ub24017/Benchmark/Zero-TIG/
nvidia-smi
python test.py --lowlight_images_path /user/work/gf19473/datasets/DID_1080 --dataset DID --model_pretrian --save /user/work/ub24017/Dataset/Zero-TIG/result/DID