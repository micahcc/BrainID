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
#include <list>

#define TIMEDIM 3
#define SLICEDIM 1
#define SECTIONDIM 0

// declare images
typedef double DataType;
typedef itk::OrientedImage<DataType, 3> Image3DType;
typedef itk::OrientedImage<DataType, 4> Image4DType;

typedef int LabelType;
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

enum { AVERAGES, LOCALMIN, LOCALMAX };

Image4DType::Pointer summ(const Image4DType::Pointer fmri_img, 
            const Label3DType::Pointer labelmap, std::list<LabelType>& sections);

Image4DType::Pointer summ(const Image4DType::Pointer fmri_img,
            std::list<int>& voxels);

/* Normalizes by the averaging each voxel over time */
Image4DType::Pointer normalizeByVoxel(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, int regions);

/* Normalizes by the averaging all the voxels in the mask/label */
Image4DType::Pointer normalizeByGlobal(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap);

/* Normalizes each region separately */
Image4DType::Pointer summRegionNorm(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, std::list<LabelType>& sections);

/* Normalizes each region by the global average */
Image4DType::Pointer summGlobalNorm(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, std::list<LabelType>& sections);

Image4DType::Pointer splitByRegion(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, int label);

Image4DType::Pointer read_dicom(std::string directory, double skip = 0);

double get_average(const Image4DType::Pointer fmri_img, 
        const Label3DType::Pointer labelmap);

Image4DType::Pointer getspline(const Image4DType::Pointer fmri_img, int sections);

Image3DType::Pointer get_average(const Image4DType::Pointer fmri_img);
Image3DType::Pointer extract(Image4DType::Pointer input, size_t index);

void removeMissing(std::list<LabelType>& ref, std::list<LabelType>& mod);
std::list<LabelType> getlabels(Label3DType::Pointer labelmap);

#endif
