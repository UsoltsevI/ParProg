#!/bin/sh
#SBATCH -n 1
#SBATCH -o ConTransferEq-%j.out # STDOUT
#SBATCH -e ConTransferEq-%j.err # STDERR

mpirun -np 1 ./contransfer "$1" "$2" $3
