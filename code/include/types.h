#ifndef TYPES_H
#define TYPES_H

/* Typedefs */
#define SERIESDIM 0
#define PARAMDIM 1
#define VARDIM 2

#define TIMEDIM 3
#define SLICEDIM 1
#define SECTIONDIM 0

//image readers
#include <itkOrientedImage.h>

//iterators
#include <itkImageLinearConstIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>

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


typedef itk::OrientedImage<double, 3> Image3DType;
typedef itk::OrientedImage<double, 4> Image4DType;

enum{TAU_0=0, ALPHA=1, E_0=2, V_0=3, TAU_S=4, TAU_F=5, EPSILON=6, A_1=7, A_2=8, PSIZE=9};

struct State
{
    double S;
    double F;
    double V;
    double Q;
};

struct Activation
{
    double time;
    double level;
};

#endif //TYPES_H
