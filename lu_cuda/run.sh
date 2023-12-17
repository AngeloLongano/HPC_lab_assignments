#!/bin/bash

# Array of dataset sizes
DATASET_SIZES=(1024 2000 4000 8000)

# Block size
BLOCK_SIZES=32

# Loop over dataset sizes
for SIZE in "${DATASET_SIZES[@]}"; do
    # Loop over block sizes
    #for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
        # Execute the make command and capture the execution time
        exec_time=$(make -s EXT_CXXFLAGS="-DCUDA_TIME -DN=${SIZE} -DBLOCK_SIZE=${BLOCK_SIZE}" clean all benchmark | tail -n 1 | cut -d ' ' -f 4)

        # Print the result
        echo "Dataset size: ${SIZE}, Block size: ${BLOCK_SIZE}, Avg execution time: ${exec_time}s" >> results.txt
    #done
done
