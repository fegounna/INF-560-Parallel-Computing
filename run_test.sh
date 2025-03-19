#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null
export OMP_NUM_THREADS=8
salloc -N 8  -c 8 -n 8 mpirun ./sobelf $OUTPUT_DIR $(ls $INPUT_DIR/*gif) 

# for i in $INPUT_DIR/*gif ; do
#     DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
#     echo "Running test on $i -> $DEST"
#     # salloc -N 1 -c 2 -n 1 mpirun ./sobelf $i $DEST

# done



