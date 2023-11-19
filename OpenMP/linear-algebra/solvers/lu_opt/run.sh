#!/bin/bash

# Verifico il num di argomenti
if [ "$#" -ne 3 ]; then
    echo "Manca uno degli argomenti, questo è l'ordine: $0 N NTHREADS COMPILER_TYPE"
    exit 1
fi

# Imposta i parametri passati dalla riga di comando
export N=$1
export OMP_NUM_THREADS=$2
COMPILER_TYPE=$3

# 1 == Polybench, 0 == Perf
if [ "$COMPILER_TYPE" -eq 1 ]; then
    echo "Using Polybench"
    make EXT_CFLAGS="-DPOLYBENCH_TIME -pg -DN=$N -DNTHREADS=$OMP_NUM_THREADS" clean all run
else
    module load perf/1.0
    echo "Using perf"
    make EXT_CFLAGS="-pg -DN=$N -DNTHREADS=$OMP_NUM_THREADS" EXT_ARGS="" clean all run
    perf stat ./lu_opt_acc
fi
