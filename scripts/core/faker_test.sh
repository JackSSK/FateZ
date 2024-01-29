#!/bin/sh
#
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256GB
#SBATCH --partition=jobs-gpu
#SBATCH --account=core-genlmu
#SBATCH --mail-user=gyu@genzentrum.lmu.de
#SBATCH --mail-type=fail
#SBATCH -o ../../logs/slurm%j.log
#SBATCH -e ../../logs/slurm%j.err
#SBATCH -J faker_test


# echo "#########"
# echo "Print the current environment (verbose)"
# env

echo "#########"
echo "Show information on nvidia device(s)"
nvidia-smi

echo "#########"
echo "Start Test"
python3 faker_test.py > ../../logs/faker_test.log
echo "#########"

# rm ../../results/faker.ckpt

