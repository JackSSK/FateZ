#!/bin/sh
#
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32GB
#SBATCH --partition=jobs-gpu
#SBATCH --account=core-genlmu
#SBATCH --mail-user=gyu@genzentrum.lmu.de
#SBATCH --mail-type=fail
#SBATCH -o logs/slurm%j.log
#SBATCH -e logs/slurm%j.err
#SBATCH -J test_gpu


# echo "#########"
# echo "Print the current environment (verbose)"
# env

echo "#########"
echo "Show information on nvidia device(s)"
nvidia-smi

echo "#########"
python3 scripts/core/faker_test.py > logs/faker_test.log
rm results/faker.ckpt

