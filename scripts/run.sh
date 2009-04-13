#!/bin/bash
mkdir "run$1"
cd "run$1"
if [[ $1 != 1 ]]
then
    ../brainid ../simseries.nii.gz ../stim.in distribution.serial ../distribution.serial
else
    ../brainid ../simseries.nii.gz ../stim.in distribution.serial
fi
ln -s distribution.serial ../distribution.serial
cd ../
