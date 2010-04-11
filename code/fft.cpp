#include <cstdio>

#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "modNiftiImageIO.h"
#include "tools.h"

#include <vcl_list.h>
#include <vul/vul_arg.h>

typedef itk::OrientedImage<double, 4> Image4DType;
typedef itk::OrientedImage<int, 3> Label3DType;

using namespace std;

int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_fmri(0 ,"Input 4D Image, to FFT");
//    vul_arg<string> a_mask(0, "Output 4D Masked Image");
    vul_arg<string> a_out(0 ,"Output 4D Image with frequencies where time was");
    
    /* Processing */
    vul_arg_parse(argc, argv);
    
    Image4DType::Pointer fmri_img;
//    Label3DType::Pointer mask_img;

    //Read Images
    {
    itk::ImageFileReader<Image4DType>::Pointer reader = 
                itk::ImageFileReader<Image4DType>::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader ->SetFileName( a_fmri() );
    fmri_img = reader->GetOutput();
    fmri_img->Update();
    }
    
//    {
//    itk::ImageFileReader<Label3DType>::Pointer reader = 
//                itk::ImageFileReader<Label3DType>::New();
//    reader->SetImageIO(itk::modNiftiImageIO::New());
//    reader->SetFileName( a_mask() );
//    mask_img = reader->GetOutput();
//    mask_img->Update();
//    }

    fprintf(stderr, "Applying FFT");
    Image4DType::Pointer out_img = fft_image(fmri_img);
    fprintf(stderr, "Done\n");
    
//    fprintf(stderr, "Applying Mask\n");
//    out_img = applymask<DataType, 4, LabelType, 3>(out_image, mask_img);
//    fprintf(stderr, "Done\n");
       
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

