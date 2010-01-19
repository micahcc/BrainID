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
    vul_arg<string> a_out(0, "Output 4D Image Spline");
    
    vul_arg<double> a_freq("-f" ,"Frequency", .25);

    vul_arg_parse(argc, argv);
    
    //Read Image
    fprintf(stderr, "Building Sine...\n");
    ImageTimeSeries::Pointer out = ImageTimeSeries::New();
    ImageTimeSeries::SizeType size = {{1, 1, 1, 1000}};
    out->SetRegions(size);
    out->Allocate();
    ImageTimeSeries::IndexType index = {{0, 0, 0, 0}};

    for(int i = 0 ; i < 1000 ; i++) {
        index[3] = i;
        out->SetPixel(index, sin(i*a_freq()*2*3.14159));
    }
    fprintf(stderr, "Done\n");
       
    fprintf(stderr, "Writing...\n");
    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetInput(out);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(a_out());
    writer->Update();
    fprintf(stderr, "Done\n");
        
    return 0;
}

