#!/bin/bash
save DICOM dir to nii.gz
bet2 3DSPGR.nii.gz brain2.nii.gz -f .4 -m
generates a skullstripped image and a mask. The skullstripped image is very basic but can be used to increase the SNR vs. the original image.
manually fix brainmask
#fslmaths file.nii.gz -kernel
unbias 3DSPGR.nii.gz 3DSPGR-unbias.nii.gz
$ fslmaths brainmask.nii.gz -kernel sphere 30 -mul 3DSPGR-unbias.nii.gz brain.nii.gz -odt double
open in slicer3 with centered checked, then save the file as brain-centered.nii.gz
#affine:
 ../../Slicer3-build/lib/Slicer3/Plugins/AffineRegistration --resampledmovingfilename affine.nii.gz --outputtransform affine.tfm brain-centered.nii.gz ~/Atlases/LPBA40/LPBA40.FLIRT.nifti/avg/lpba40.flirt.avg152T1_brain.brain.avg.nii.gz --echo  &> affine.out
 #options:
    FixedImageSmoothingFactor: 0
    MovingImageSmoothingFactor: 0
    HistogramBins: 30
    SpatialSamples: 10000
    Iterations: 2000
    TranslationScale: 100
    InitialTransform: 
    FixedImageFileName: brain.nii.gz
    MovingImageFileName: /home/micahcc/Atlases/LPBA40/LPBA40.FLIRT.nifti/avg/lpba40.flirt.avg152T1_brain.brain.avg.nii.gz
    OutputTransform: affine.tfm
    ResampledImageFileName: affine.nii.gz
    echoSwitch: 1
    xmlSwitch: 0
    processInformationAddressString: 0
#bspline:
 ../../Slicer3-build/lib/Slicer3/Plugins/BSplineDeformableRegistration --initialtransform affine.tfm --outputtransform bspline.tfm --resampledmovingfilename bspline.nii.gz brain.nii.gz ~/Atlases/LPBA40/LPBA40.FLIRT.nifti/avg/lpba40.flirt.avg152T1_brain.brain.avg.nii.gz  &> bspline.out
Command Line Arguments
    Iterations: 20
    gridSize: 5
    HistogramBins: 100
    SpatialSamples: 50000
    ConstrainDeformation: 0
    MaximumDeformation: 1
    DefaultPixelValue: 0
    InitialTransform: affine.tfm
    FixedImageFileName: brain.nii.gz
    MovingImageFileName: /home/micahcc/Atlases/LPBA40/LPBA40.FLIRT.nifti/avg/lpba40.flirt.avg152T1_brain.brain.avg.nii.gz
    OutputTransform: bspline.tfm
    ResampledImageFileName: bspline.nii.gz
    echoSwitch: 1
    xmlSwitch: 0
    processInformationAddressString: 0

#applying transforms
~/Slicer3-build/lib/Slicer3/Plugins/ResampleVolume2 -f bspline.tfm -n -R brain.nii.gz -z 256,512,256 ~/Atlases/LPBA40/LPBA40.FLIRT.nifti/avg/lpba40.flirt.avg152T1_brain.brain.avg.nii bsplinetest.nii.gz 
apply -z as necessary to prevent clipping
~/Slicer3-build/lib/Slicer3/Plugins/ResampleVolume2 -f bspline.tfm -n 4 -R brain.nii.gz -z 256,512,256 ~/Atlases/LPBA40/LPBA40.FLIRT.nifti/avg/csf.nii.gz csf.nii.gz
#labelmap:
~/Slicer3-build/lib/Slicer3/Plugins/ResampleVolume2 -f bspline.tfm -R brain.nii.gz -i nn -n 4 -z 256,512,256  ~/Atlases/LPBA40/LPBA40.FLIRT.nifti/maxprob/lpba40.flirt.avg152T1_brain.label.nii.gz label.nii.gz
#-z makes the images larger which helps with clipping

#now we have  probability maps for csf, wm, gm
#to get the probability map of being background:
#first, need to normalize csf,wm,gm, the following gives the max value
fslstats csf.nii.gz -R
#then divide
fslmaths csf.nii.gz -div $MAX csf_norm.nii.gz -odt float

#to get the background probability, 
fslmaths csf.nii.gz -kdd gm.nii.gz -add wm.nii.gz sum.nii.gz -odt float
fslstats sum.nii.gz -R
fslmaths sum.nii.gz -div $MAX fg.nii.gz -odt float
fslmaths fg.nii.gz -sub 1 -abs bg.nii.gz

#use emsegment in slicer3
#to extract masks for gm,wm and csf use the following
fslmaths segments.nii.gz -thr 127 -uthr 127 -bin gm_mask.nii.gz

#to deal with patient moving head or other coordinate system strangeness
#to do this, one needs to acquire a timeslice of an fmri image
../../Slicer3-build/lib/Slicer3/Plugins/RigidRegistration --resampledmovingfilename rigid_to_fmri.tfm fmri.nii.gz brain.nii.gz
~/Slicer3-build/lib/Slicer3/Plugins/ResampleVolume2 -t rt -f rigid_to_fmri.tfm -R brain.nii.gz -i nn -n 4 -z 256,512,256 gm_mask.nii.gz gm_mask_final.nii.gz
~/Slicer3-build/lib/Slicer3/Plugins/ResampleVolume2 -t rt -f rigid_to_fmri.tfm -R brain.nii.gz -i nn -n 4 -z 256,512,256 label.nii.gz label_final.nii.gz
