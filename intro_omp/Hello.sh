#!/bin/sh
#SBATCH -J hello_omp
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o hello_omp-%j.out
#SBATCH -e hello_omp-%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

./hello "$1"