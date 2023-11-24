# High performance computing - Assignment 1 (OpenMP) - Group 9

## Team members

name | student ID | email
---- | ---------- | -----
Angelo Longano | 153055 | 285288@studenti.unimore.it
Euplio Lisi | 191037 | 245826@studenti.unimore.it
Riccardo Cracco | 192326 | 281925@studenti.unimore.it

## LU decomposition

Our assignment was to improve performances of matrix [LU decomposition](https://en.wikipedia.org/w/index.php?title=LU_decomposition&oldid=1185770152).

Original Polybench's version of this algorithm is in `OpenMP/linear-algebra/solvers/lu/lu.c`. From this version, we have applied different combinations of OpenMP's directives in order to improve performance. Each one of these versions is in a different `lu_optX.c` (where `X` is a number) in the `OpenMP/linear-algebra/solvers/lu` directory.

## How to run a single version

Execute the `OpenMP/linear-algebra/solvers/lu/run.sh` script by providing these positional parameters:
1. the name of the version to run (which is the name of the C file WITHOUT the `.c` extension, e.g. `lu` to run the original version);
2. the dataset size to use. Must be one between `MINI_DATASET`, `SMALL_DATASET`, `STANDARD_DATASET`, `LARGE_DATASET`, `EXTRALARGE_DATASET` (these dataset sizes are defined in `OpenMP/linear-algebra/solvers/lu/lu.h`);
3. the number of threads to use (default 4);
4. the back-end to use to collect statistics about program's execution. Must be one between `none` (only the Polybench time of a **single** run will be printed), `perf`, `polybench` (will use the `OpenMP/utilities/time_benchmark.sh` script to take the average execution time of 5 runs), `gprof`.

Example:

``` bash
./run.sh lu_opt1 EXTRALARGE_DATASET 4 polybench
```

## Generate statistics for **all** versions

Run the `OpenMP/linear-algebra/solvers/lu/stats.sh` script.

This script will generated a `statistics.txt` file in the `OpenMP/linear-algebra/solvers/lu` directory with the average execution time of 5 runs for all versions, for all dataset sizes.

Example of `statistics.txt`:

```
TEST lu
size MINI_DATASET
0.00002266
size SMALL_DATASET
0.00135266
size STANDARD_DATASET
0.97247966
size LARGE_DATASET
6.89780700
size EXTRALARGE_DATASET
55.61338433

TEST lu_opt1
size MINI_DATASET
0.00010433
size SMALL_DATASET
0.00074966
size STANDARD_DATASET
0.59216400
size LARGE_DATASET
4.61016400
size EXTRALARGE_DATASET
38.25809833
```

## Check correctness of the algorithm

Run the `OpenMP/linear-algebra/solvers/lu/check-output.sh` to check that all optimized versions produce the same output (for all available dataset sizes) as the original, sequential, version.   

