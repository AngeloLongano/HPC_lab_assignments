module load perf/1.0

echo "base"
make EXT_CFLAGS="-pg" EXT_ARGS="" clean all run
perf stat ./lu_acc



