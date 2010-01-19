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

int main()
{
    /* Input related */
    vul_arg<string> a_input(0 ,"Image to Calculate RMS of");
    vul_arg<string> a_output(0 ,"Output RMS Image");
    
    vul_arg_parse(argc, argv);
    
    itk::ImageFileReader<Image4DType>::Pointer reader = 
                itk::ImageFileReader<Image4DType>::New();
    reader->SetFileName( a_input() );
    labelmap_img->Update();
    get_rms(reader->GetOutput(), 
}
