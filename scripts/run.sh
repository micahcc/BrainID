#!/bin/bash
mkdir "run$1"
cd "run$1"
PROCESSES=4
if [[ $1 != 1 ]]
then
    pwd
    echo time mpirun -c $PROCESSES ../brainid -t ../simseries.nii.gz -s ../stim.in --serialout distribution.serial --serialin ../distribution.serial -p 30000
    time mpirun -c $PROCESSES ../brainid -t ../simseries.nii.gz -s ../stim.in --serialout distribution.serial --serialin ../distribution.serial -p 30000
else
    echo time mpirun -c $PROCESSES ../brainid -t ../simseries.nii.gz -s ../stim.in --serialout distribution.serial -p 30000
    time mpirun -c $PROCESSES ../brainid -t ../simseries.nii.gz -s ../stim.in --serialout distribution.serial -p 30000
fi
rm "../distribution.serial"
ln -s "$(pwd)/distribution.serial" "../distribution.serial"
cd ../
