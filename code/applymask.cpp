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
    vul_arg<string> a_mask(0 ,"Input Mask Image");
    vul_arg<string> a_out(0, "Output 4D Masked Image");
    
    /* Processing */
    vul_arg_parse(argc, argv);
    
    Image4DType::Pointer fmri_img, out_img;
    Label3DType::Pointer mask_img;

    //Read Image
    {
    itk::ImageFileReader<Image4DType>::Pointer reader = 
                itk::ImageFileReader<Image4DType>::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader ->SetFileName( a_fmri() );
    fmri_img = reader->GetOutput();
    fmri_img->Update();
    }
    
    {
    itk::ImageFileReader<Label3DType>::Pointer reader = 
                itk::ImageFileReader<Label3DType>::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_mask() );
    mask_img = reader->GetOutput();
    mask_img->Update();
    }
    
    fprintf(stderr, "Applying Mask\n");
    out_img = applymask<DataType, 4, LabelType, 3>(fmri_img, mask_img);
    fprintf(stderr, "Done\n");
       
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

