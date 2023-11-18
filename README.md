# Assignment 1 (solver lu) Group 9 

## Directory original lu.c

```
    cd OpenMP/linear-algebra/solvers/lu
```

## Directory optimized lu.c

```
    cd OpenMP/linear-algebra/solvers/lu_opt
```

## Run application

```
    ./run.sh N_SIZE N_THREADS PERF_OR_POLYBENCH
```
N_SIZE --> from 1024 at 8192 tested
N_THREADS --> from 1 to 4
PERF_OR_POLYBENCH --> 0 with perf, 1 with polybench

Examples
#### Test on original code
./run.sh 1024 1 1
./run.sh 2048 1 1
./run.sh 4096 1 1
./run.sh 8192 1 1

#### Test on optimized code
./run.sh 1024 4 1
./run.sh 2048 4 1
./run.sh 4096 4 1
./run.sh 8192 4 1