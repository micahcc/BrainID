#!/bin/bash

BREAK=4
STOP=`printf '%04d' 100`
START=`printf '%04d' 0`
DIV=1024
WEIGHT=1
PARTICLES=200000
EXPWEIGHT=1
time mpirun -c 4  ../../brainid -tstop $STOP -log brainid$START.log \
            -ts noise-bold.nii.gz -div $DIV -stim stimuli -p $PARTICLES \
            -so $STOP.serial -weightvar $WEIGHT -expweight $EXPWEIGHT \
            -yo particle-bold-$START.nii.gz -xo particle-state-$START.nii.gz

../../concatinate particle-bold.nii.gz particle-bold-*
../../concatinate particle-state.nii.gz particle-state-*
../plotout.py --prefix "out$START"
for i in `seq 1 $BREAK`; do
    STOP=`printf '%04d' $(( (i+1)*100 ))`
    START=`printf '%04d' $(( i*100 ))`
    time mpirun -c 4  ../../brainid -tstop $STOP -log brainid$START.log \
                -ts noise-bold.nii.gz -div $DIV -stim stimuli -p $PARTICLES \
                -si $START.serial -so $STOP.serial -weightvar $WEIGHT -expweight $EXPWEIGHT \
                -yo particle-bold-$START.nii.gz -xo particle-state-$START.nii.gz -weightvar $WEIGHT
    rm particle-bold.nii.gz
    rm particle-state.nii.gz
    ../../concatinate particle-bold.nii.gz particle-bold-*
    ../../concatinate particle-state.nii.gz particle-state-*
    ../plotout.py --prefix "out$START"
done
