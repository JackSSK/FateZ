#!/bin/sh
#
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32GB
#SBATCH --partition=jobs-gpu
#SBATCH --account=core-genlmu
#SBATCH --mail-user=gyu@genzentrum.lmu.de
#SBATCH --mail-type=fail
#SBATCH -o ../../logs/slurm%j.log
#SBATCH -e ../../logs/slurm%j.err
#SBATCH -J test_gpu


echo "#########"
python3 make_data_loader.py > ../../logs/dl_test.log


