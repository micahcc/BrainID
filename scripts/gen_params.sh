#!/bin/bash

for i in `seq 1 50`; do printf '%s %s\n' `~/brainid/build/code/boldgen -end 1 2> /dev/null | awk '{ print $1, $2, $3, $4, $5, $6, $7}'` 4.2 1 | xargs >> params;  done 
