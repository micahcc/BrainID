#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"
#include "tools.h"

#include <itkMultiplicationImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkRandomImageSource.h>
#include <itkExtractImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include "modNiftiImageIO.h"

#include <sstream>
#include <iostream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

typedef itk::OrientedImage<double, 4> ImageTimeSeries;

using namespace std;

Image3DType::Pointer createRegions(Image4DType::Pointer templ = NULL)
{
    Image3DType::Pointer templ3d;

    itk::RandomImageSource<Image3DType>::Pointer rImage = 
                itk::RandomImageSource<Image3DType>::New();
    rImage->SetNumberOfThreads(10);
    rImage->SetMax(1);
    rImage->SetMin(0);
    itk::BinaryThresholdImageFilter<Image3DType>::Pointer thresh = 
                itk::BinaryThresholdImageFilter<Image3DType>::New();

    if(templ) {
        /* Extract a volume at timemax/2 */
        Image4DType::RegionType region = templ->GetLargestPossibleRegion();
        region.GetIndex()[3] = region.GetSize()[3]/2;
        region.GetSize()[3] = 0;
        itk::ExtractImageFilter<Image4DType, Image3DType>::Pointer extractF =
                    itk::ExtractImageFilter<Image4DType, Image3DType>::New();
        extractF->SetInput(templ);
        extractF->SetExtractionRegion(region);
    
        thresh->SetInput(extractF->GetOutput());
        thresh->SetLowerThreshold(-.1);
        thresh->SetUpperThreshold(.1);
        thresh->SetInsideValue(0);
        thresh->SetOutsideValue(1);
        thresh->Update();

        templ3d = thresh->GetOutput();
    } else {
        Image4DType::SizeType out_size;
        out_size[0] = 30;
        out_size[1] = 30;
        out_size[2] = 30;
        out_size[3] = 30; 
        
        //outimage
        rImage->SetRegions(out_size);
    }
    
    /* Create a Random Field, Perform Smoothing to the set number of resels
     * then threshold, to create the regions
     */
    itk::DiscreteGaussianImageFilter<Image3DType>::Pointer gaussF = 
                itk::DiscreteGaussianImageFilter<Image3DType>::New();
    double variance[3] = {6,6,6};
    gaussF->SetVariance(variance);
    gaussF->SetInput(rImage->GetOutput());

    thresh->SetInput(gaussF->GetOutput());
    thresh->SetLowerThreshold(0);
    thresh->SetUpperThreshold(.95);
    thresh->SetInsideValue(0);
    thresh->SetOutsideValue(1);
    
    itk::MultiplicationImageFilter<Image3DType>::Pointer multiF = 
                itk::MultiplicationImageFilter<Image3DType>::New();
    multiF->SetFirstInput(tresh->GetOutput());
    multiF->SetSecondInput(templ3d);
    multiF->Update();

    return multiF->GetOutput();
}

Image4DType::Pointer simulate(string param_f, string act_f, 
            Image3DType::Pointer regions)
{
    //segment image into discrete blobs
    
    //open and read param_f
    //select a number of regions matching param_f parameter sets
    //to develop
    
    //open act_f

    //simulate each region, using act_f for activation times

}

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_templ("-t" ,"Template (takes orientation/spacing/brain)");
    vul_arg<string> a_active("-a" ,"Activation file: <onset> <scale>");
    vul_arg<string> a_imageOut("-o" ,"Image Out");
    vul_arg<int> a_regionSize("-s" ,"Average Size of regions to simulate");
    vul_arg<string> a_params("-p" ,"File with parameters, 1 region per line,"
                    " space separated: TAU_S TAU_F EPSILON TAU_0 ALPHA E_0 V_0"); 
    
    /* Processing */
    vul_arg_parse(argc, argv);
    
    Image4DType::Pointer templ;
    Image4DType::Pointer out = Image4DType::New();
    
    /* Output Image */
    if(a_templ() != "") {
        itk::ImageFileReader<Image4DType>::Pointer reader = 
                    itk::ImageFileReader<Image4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader ->SetFileName( a_templ() );
        templ = reader->GetOutput();
        templ->Update();
    
    /* Decide on activation positions/sizes */
    if(templ) {
        
    }

    
    fprintf(stderr, "Writing...\n");
    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetInput(out_img);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(a_out());
    writer->Update();
    fprintf(stderr, "Done\n");
        
    return 0;
}

