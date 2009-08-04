#ifndef SEGMENT_H
#define SEGMENT_H

//image readers
#include <itkOrientedImage.h>

//test
#include <itkImageConstIteratorWithIndex.h>

//iterators
#include <itkImageLinearConstIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>

//standard libraries
#include <vector>

// declare images
typedef double DataType;
typedef itk::OrientedImage<DataType, 3> Image3DType;
typedef itk::OrientedImage<DataType, 4> Image4DType;

typedef short LabelType;
typedef itk::OrientedImage<LabelType, 3> Label3DType;
typedef itk::OrientedImage<LabelType, 4> Label4DType;

typedef itk::ImageSliceIteratorWithIndex< Image4DType > SliceIterator4D;
typedef itk::ImageLinearIteratorWithIndex< Image4DType > PixelIterator4D;
typedef itk::ImageLinearIteratorWithIndex< Image3DType > PixelIterator3D;

//Each SectionType struct contains an integer label
//and an iterator that moves forward in time
struct SectionType{
    int label;
    PixelIterator4D point;
} ;

/* Normalizes by the averaging each voxel over time */
Image4DType::Pointer normalizeByVoxel(const Image4DType::Pointer fmri_img);

/* Normalizes by the averaging all the voxels in the mask/label */
Image4DType::Pointer normalizeByGlobal(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap);

/* Normalizes by the averaging all the voxels in the mask/label */
Image4DType::Pointer normalizeByRegion(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap);

std::vector< Image4DType::Pointer > splitByRegion(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap);

Image4DType::Pointer applymask(const Image4DType::Pointer fmri_img, 
            const Label3DType::Pointer mask_img);

Image4DType::Pointer read_dicom(std::string directory, double skip = 0);

#endif
