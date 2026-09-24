#!/bin/sh
#SBATCH -J sum_omp
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o sum_omp-%j.out
#SBATCH -e sum_omp-%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

./sum "$1" "$2"