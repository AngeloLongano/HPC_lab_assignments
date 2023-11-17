module load perf/1.0

echo "optimized"
make EXT_CFLAGS="-pg" EXT_ARGS="" clean all run
perf stat ./lu_opt_acc
