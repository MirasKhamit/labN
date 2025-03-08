#!/bin/bash

# Define the array sizes and block sizes
array_sizes=(10000 11000 12000 13000 14000)
block_sizes=(32 64 128 256 512)
# Compile
nvcc labN.cu
# Loop through the array sizes
for array_size in "${array_sizes[@]}"; do
    # Loop through the block sizes
    for block_size in "${block_sizes[@]}"; do
        echo "Running for ArraySize=$array_size, BlockSize=$block_size"
        
        # Mode 1: Naive GEMV
        ncu --set full ./a.out  $array_size $block_size 1 > labNOutput/naive$array_size=$block_size.txt
        # Mode 2: Shared Memory Optimized GEMV
        ncu --set full ./a.out  $array_size $block_size 2 > labNOutput/shared$array_size=$block_size.txt
        # Mode 3: Global Memory Optimized GEMV
        ncu --set full ./a.out  $array_size $block_size 3 > labNOutput/global$array_size=$block_size.txt
        # Mode 4: Register Optimized GEMV
        ncu --set full ./a.out  $array_size $block_size 4 > labNOutput/registers$array_size=$block_size.txt

    done
done


echo "All iterations completed."