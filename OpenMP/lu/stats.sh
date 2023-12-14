#!/bin/bash

NAME_FILE_RESULTS="${1:-"statistics.txt"}"
echo "" > $NAME_FILE_RESULTS

# Array per memorizzare i tempi di lu_opt1 per ciascun dataset
declare -A lu_opt1_times

for file in *.c; do
    name="${file%.c}"
    echo "Generating statistics for version ${name}..."
    echo "TEST $name" >> $NAME_FILE_RESULTS

    for size in MINI_DATASET SMALL_DATASET STANDARD_DATASET LARGE_DATASET EXTRALARGE_DATASET; do
        echo "dataset size = ${size}..."

        time=$(./run.sh $name $size 4 polybench | tail -n 1 | cut -d ' ' -f 4)

        # Aggiorna il tempo migliore se il tempo corrente è minore
        if [[ "$name" == "lu_opt1" ]]; then
            lu_opt1_times["$size"]=$time
            echo "$time --------------- Best solution " >> $NAME_FILE_RESULTS
        else
            if [[ "$name" == "lu" ]]; then
                echo "$time --------------- Not optimized solution" >> $NAME_FILE_RESULTS
            else
                # Calcola e salva la differenza
                if [[ "$name" != "lu" ]]; then
		diff=$(echo "$time - ${lu_opt1_times[$size]}" | bc)
                # Stampa il tempo e la differenza
                echo "$time --------------- Time difference with the best solution: $diff" >> $NAME_FILE_RESULTS
           fi   
       		fi
        fi
    done

    printf "\n" >> $NAME_FILE_RESULTS
    echo "----------" >> $NAME_FILE_RESULTS
done

