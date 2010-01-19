#!/bin/bash

fslmaths $1 -tmean -sqr meansqr.nii.gz
fslmaths $1 -tvar -add meansqr.nii.gz -sqrt rms.nii.gz
