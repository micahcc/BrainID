//image readers
#include "itkOrientedImage.h"

//test
#include "itkImageConstIteratorWithIndex.h"

//iterators
#include "itkImageLinearConstIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"

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
typedef struct {
    int label;
    SliceIterator4D point;
} SectionType ;

SectionType findLabel(std::list<SectionType>& list, int label);

//sort_voxels fills the list given with new SectionType structs, each of 
//which represents a label from the labelmap image. It then finds each
//member voxel of each label and fills the list in the SectionType
//with iterators for the member voxels.
void sort_voxels(const Image4DType::Pointer fmri_img, 
            const Image3DType::Pointer label_img,
            std::list<SectionType>& voxels);
//should be called when you are done using voxels_list
void free_voxels(std::list<SectionType>& voxels_list);

Image4DType::Pointer read_dicom(std::string directory);
