#!/bin/bash

# run all 40 splits of MINT dataset
# drag this file to same directory as signet.py before running
for i in `seq 1 40`; do
    python signet.py --dataset=mint --split=$i;
done