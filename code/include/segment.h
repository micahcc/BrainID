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
#include <list>

// declare images
typedef signed short PixelType;
typedef itk::OrientedImage<PixelType, 3> Image3DType;
typedef itk::OrientedImage<PixelType, 4> Image4DType;

typedef itk::ImageSliceIteratorWithIndex< Image4DType > SliceIterator4D;
typedef itk::ImageLinearIteratorWithIndex< Image4DType > PixelIterator4D;
typedef itk::ImageLinearIteratorWithIndex< Image3DType > PixelIterator3D;

//Each SectionType struct contains an integer label and a list of SliceIterators
//as well a pointer to the original image, just for completeness (whats 8 bytes 
//among friends)
struct SectionType{
    int label;
    PixelIterator4D point;
} ;

//sort_voxels fills the list given with new SectionType structs, each of 
//which represents a label from the labelmap image. It then finds each
//member voxel of each label and fills the list in the SectionType
//with iterators for the member voxels.
void segment(const Image4DType::Pointer fmri_img, 
            const Image3DType::Pointer label_img,
            std::list<SectionType>& voxels);
//should be called when you are done using voxels_list

Image4DType::Pointer read_dicom(std::string directory);

#endif
