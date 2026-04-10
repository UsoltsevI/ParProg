#!/bin/sh
#SBATCH -o SzTransferEq-%j.out
#SBATCH -e SzTransferEq-%j.err


NP=$SLURM_NTASKS
K=$1
M=$2
OUT=$3

echo "Running with NP=$NP, K=$K, M=$M, out=${OUT:-'(none)'}"

if [ -n "$OUT" ]; then
    mpirun -np $NP ./sztransfer $K $M $OUT
else
    mpirun -np $NP ./sztransfer $K $M
fi
