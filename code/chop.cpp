#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "segment.h"
#include "tools.h"

#include <itkRegionOfInterestImageFilter.h>

#include <sstream>
#include <iostream>
#include <vector>

#include <vul/vul_arg.h>

using namespace std;

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_input(0 ,"Input Image");
    vul_arg<string> a_prefix(0 ,"Output Prefix");
    vul_arg<int> a_num("-n" ,"Number of slices (all chopping is done by z index)", 10);
   
    /* Processing */
    vul_arg_parse(argc, argv);
    
    Image4DType::Pointer input;
    Image4DType::RegionType region, outregion;
    Image4DType::Pointer output;
    Image4DType::IndexType index;
    Image4DType::SizeType size;

    itk::RegionOfInterestImageFilter<Image4DType, Image4DType >::Pointer extract = 
                itk::RegionOfInterestImageFilter<Image4DType, Image4DType >::New();
    
    {
        itk::ImageFileReader<Image4DType>::Pointer reader = 
                    itk::ImageFileReader<Image4DType>::New();
        reader->SetFileName( a_input() );
        reader->Update();
        input = reader->GetOutput();
    }
    extract->SetInput(input);
    region = input->GetRequestedRegion();
    index = region.GetIndex();
    size = region.GetSize();
    for(int i = 0 ;i < a_num() ; i++) {
        index[2] = i*region.GetSize()[2]/a_num();
        size[2] = (i+1)*region.GetSize()[2]/a_num() - index[2];
        outregion.SetSize(size);
        outregion.SetIndex(index);
        extract->SetRegionOfInterest(outregion);
        extract->Update();
        {
            std::ostringstream oss;
            oss << a_prefix() <<  setfill('0') << setw(3) << i << ".nii.gz";
            itk::ImageFileWriter<Image4DType>::Pointer writer = 
                        itk::ImageFileWriter<Image4DType>::New();
            writer->SetFileName( oss.str() );
            writer->SetInput(extract->GetOutput());
            std::cout << "Writing " << oss.str() << std::endl;
            writer->Update();
        }
    }
    
    return 0;
}


