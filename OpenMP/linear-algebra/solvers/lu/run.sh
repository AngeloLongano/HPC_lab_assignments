#!/bin/bash
# module load perf/1.0

# possible dataset_size: MINI_DATASET, SMALL_DATASET, STANDARD_DATASET, LARGE_DATASET, EXTRALARGE_DATASET
# possible STATISTICS: polybench, perf

# ./run.sh DATA_SIZE N_THREADS STATISTICS

NAME_FILE="${1:-"lu_opt1"}"
DATA_SIZE="${2:-"STANDARD_DATASET"}"
N_THREADS="${3:-"4"}"
STATISTICS="${4:-"polybench"}"

echo "-------------------------------------"
echo "Running $NAME_FILE with $DATA_SIZE dataset and $N_THREADS threads ($STATISTICS)"
echo "-------------------------------------"
make EXT_CFLAGS="-pg -DPOLYBENCH_TIME -D$DATA_SIZE -DNTHREADS=$N_THREADS" EXT_ARGS="" BENCHMARK=$NAME_FILE clean all run

case $STATISTICS in
    none)
        ;;
    perf)
        echo "Using perf"
        perf stat ./lu_acc
        ;;
    polybench)
        echo "Using polybench benchmark"
        make benchmark
        ;;
    gprof)
        echo "Using gprof"
        NO_OPT="-O0 -g -fopenmp"
        make EXT_CFLAGS="-pg -D$DATA_SIZE -DNTHREADS=$N_THREADS" EXT_ARGS="" BENCHMARK=$NAME_FILE OPT=$NO_OPT clean all run
        gprof lu_acc gmon.out > analysis.txt
esac