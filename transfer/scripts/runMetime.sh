#!/bin/bash
#SBATCH -o Metime-%j.out
#SBATCH -e Metime-%j.err

NP=$SLURM_NTASKS

mpirun -np $NP ./metime
