#!/usr/bin/env bash

set -e

# Checks that the parallelized versions produce the same output as the sequential version

DATASET_SIZES="MINI_DATASET SMALL_DATASET STANDARD_DATASET LARGE_DATASET EXTRALARGE_DATASET"
UNOPTIMIZED_VERSION_NAME="lu"

# use DATA_TYPE = int to avoid false positive results due to floating point arithmetics
EXT_CFLAGS="-DPOLYBENCH_DUMP_ARRAYS -DDATA_TYPE=int -DDATA_PRINTF_MODIFIER='\"%d \"'"

# Generate the output (if not exist) for the sequential version
for n in $DATASET_SIZES; do
    output_file_name="output.${UNOPTIMIZED_VERSION_NAME}.${n}.txt"

    if [ ! -f "${output_file_name}" ]; then
        make --silent EXT_CFLAGS="-D${n} ${EXT_CFLAGS}" clean all run 2> "${output_file_name}"
        echo "Generated output file for sequential version for data size = ${n}."
    fi
done

# Generate the output for the optimized versions and compare with the sequential version
for v in lu_opt*.c; do
    echo "Checking output of optimized version ${v%.c}..."

    for n in $DATASET_SIZES; do
	output_file_name="output.${v}.${n}.txt"
        make --silent EXT_CFLAGS="-D${n} ${EXT_CFLAGS}" clean all run 2> "${output_file_name}"
        
	if ! diff --brief --ignore-space-change "output.${UNOPTIMIZED_VERSION_NAME}.${n}.txt" "output.${v}.${n}.txt" > /dev/null; then
            echo "ERROR! Optimized version ${v} output is different from unoptimized version for data set size = ${n}." >&2
	    exit 1
	else
            echo "Optimized version ${v} is correct for data set size = ${n}."
	fi
    done

    echo "Finished checking for optimized version ${v}."
done

echo "Check finished."
echo "All optimized versions generate the same output as the sequential version."

exit 0

