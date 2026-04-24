#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o integrate-%j.out
# STDOUT
#SBATCH -e integrate-%j.err
# STDERR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./integrate