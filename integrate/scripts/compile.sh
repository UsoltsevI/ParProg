#!/bin/sh
gcc -fopenmp integrate.c -o integrate -lm -I$HOME/local/include -L$HOME/local/lib -lmpfr