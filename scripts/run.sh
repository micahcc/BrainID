#!/bin/bash
mkdir "run$1"
cd "run$1"
if [[ $1 != 1 ]]
then
    pwd
    echo time ../brainid ../simseries.nii.gz ../stim.in distribution.serial ../distribution.serial
    time ../brainid ../simseries.nii.gz ../stim.in distribution.serial ../distribution.serial
else
    echo time ../brainid ../simseries.nii.gz ../stim.in distribution.serial
    time ../brainid ../simseries.nii.gz ../stim.in distribution.serial
fi
rm "../distribution.serial"
ln -s "$(pwd)/distribution.serial" "../distribution.serial"
cd ../
