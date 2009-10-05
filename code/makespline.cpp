#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"

#include <itkMaskImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include "modNiftiImageIO.h"

#include <sstream>
#include <iostream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

typedef itk::OrientedImage<double, 4> ImageTimeSeries;

using namespace std;

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_fmri(0 ,"Input 4D Image");
    vul_arg<string> a_out(0, "Output 4D Image Spline");
    
    /* Processing */
    vul_arg<unsigned int> a_spline("-r" ,"Number of regions to use with spline"
                " detrending, this is the number of knots minus 2, since a beginning"
                " knot and ending knot are also used", 3.);

    vul_arg_parse(argc, argv);
    
    //Read Image
    itk::ImageFileReader<Image4DType>::Pointer reader = 
                itk::ImageFileReader<Image4DType>::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader ->SetFileName( a_fmri() );
    Image4DType::Pointer fmri_img = reader->GetOutput();
    fmri_img->Update();

    fprintf(stderr, "Building Spline...\n");
    Image4DType::Pointer spline = getspline(fmri_img, a_spline());
    fprintf(stderr, "Done\n");
       
    fprintf(stderr, "Writing...\n");
    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetInput(spline);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(a_out());
    writer->Update();
    fprintf(stderr, "Done\n");
        
    return 0;
}

