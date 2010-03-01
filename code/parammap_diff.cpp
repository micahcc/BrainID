#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMetaDataObject.h>

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include "tools.h"
#include "modNiftiImageIO.h"

#include <cmath>
#include <iostream>
#include <string>

#include <vul/vul_arg.h>

using namespace std;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;
            
/* Main Function */
int main(int argc, char* argv[])
{
    vul_arg<string> a_params1(0, "4D param file, should be smaller image, "
                "in the order: TAU_0, ALPHA, E_0, V_0, TAU_S, TAU_F, EPSILON");
    vul_arg<string> a_params2(0, "4D param file, in the order: TAU_0, ALPHA,"
                "E_0, V_0, TAU_S, TAU_F, EPSILON");
    vul_arg<string> a_output(0, "4D Output Image with percent diff");
    
    vul_arg<bool> a_invert("-i", "Show 1/(percent difference), thus bright "
                "spots will be good, and dark spots are bad", false);
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("No Warning, just echoing");

    Image4DType::Pointer paramImage1;
    Image4DType::Pointer paramImage2;
    Image4DType::Pointer output;

    /* Open up the input */
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_params1() );
    reader->Update();
    paramImage1 = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_params1().c_str());
        exit(-1);
    }
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_params2() );
    reader->Update();
    paramImage2 = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_params2().c_str());
        exit(-1);
    }
    
    output = pctDiff(paramImage1, paramImage2);
    output->CopyInformation(paramImage1);
    
    //write final output
    {
    itk::ImageFileWriter<Image4DType>::Pointer out = 
        itk::ImageFileWriter<Image4DType>::New();
    out->SetInput(output);
    out->SetFileName(a_output());
    cout << "Writing: " << a_output() << endl;
    out->Update();
    }

    return 0;

}




