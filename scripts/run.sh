#!/bin/bash
mkdir "run$1"
STIM="stim.in"
SERIES="simseries.nii.gz"
PARTICLES=60000
cp  ../../code/* "run$1"
cp  ../../code/include/* "run$1"
cp "$SERIES" "run$1"
cp "$0" "run$1"
cp "$STIM" "run$1"
cp sim*out "run$1"
cp plotout.m "run$1"
cd "run$1"
PROCESSES=3
if [[ $1 != 1 ]]
then
    pwd
    echo time mpirun -c $PROCESSES ../brainid -t "$SERIES" -s "$STIM" --serialout distribution.serial --serialin distribution.serial -p "$PARTICLES"
    time mpirun -c $PROCESSES ../brainid -t "$SERIES" -s "$STIM" --serialout distribution.serial --serialin ../distribution.serial -p "$PARTICLES"
else
    echo time mpirun -c $PROCESSES ../brainid -t "$SERIES" -s "$STIM" --serialout distribution.serial -p "$PARTICLES"
    time mpirun -c $PROCESSES ../brainid -t "$SERIES" -s "$STIM" --serialout distribution.serial -p "$PARTICLES"
fi
rm "../distribution.serial"
ln -s "$(pwd)/distribution.serial" "../distribution.serial"
cd ../
