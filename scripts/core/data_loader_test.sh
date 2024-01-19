#!/bin/sh
#
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=jobs-cpu-long
#SBATCH --account=core-genlmu
#SBATCH --mail-user=gyu@genzentrum.lmu.de
#SBATCH --mail-type=fail
#SBATCH -o ../../logs/slurm%j.log
#SBATCH -e ../../logs/slurm%j.err
#SBATCH -J integrate_corpus

python3 make_data_loader.py > ../../logs/dl_test_test.log
