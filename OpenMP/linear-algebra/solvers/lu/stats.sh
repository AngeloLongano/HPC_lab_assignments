NAME_FILE_RESULTS="${1:-"statistics.txt"}"
echo "" > $NAME_FILE_RESULTS

for file in *.c; do
    name="${file%.c}"
    echo "Generating statistics for version ${name}..."
    echo "TEST $name" >> $NAME_FILE_RESULTS
    for size in MINI_DATASET SMALL_DATASET STANDARD_DATASET LARGE_DATASET EXTRALARGE_DATASET; do
	echo "dataset size = ${size}..."
        echo "size $size" >> $NAME_FILE_RESULTS
        ./run.sh $name $size 4 polybench | tail -n 1 | cut -d ' ' -f 4 >> $NAME_FILE_RESULTS
    done
    printf "\n" >> $NAME_FILE_RESULTS
    echo "----------"
done
