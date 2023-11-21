NAME_FILE_RESULTS="${1:-"statistics.txt"}"

for file in *.c; do
    name="${file%.c}"
    echo "TEST $name" >> $NAME_FILE_RESULTS
    for size in MINI_DATASET SMALL_DATASET STANDARD_DATASET LARGE_DATASET EXTRALARGE_DATASET; do
        echo "size $size" >> $NAME_FILE_RESULTS
        ./run.sh $name $size 4 perf | tail -n 1 >> $NAME_FILE_RESULTS
    done
    printf "\n" >> $NAME_FILE_RESULTS
done