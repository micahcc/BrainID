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

namespace aux = indii::ml::aux;

/* Main Function */
int main(int argc, char* argv[])
{
    vul_arg<string> a_input1(0, "4D param file timeseries file");
    vul_arg<string> a_input2(0, "4D param file timeseries file");
    vul_arg<string> a_output(0, "3D Output Image with MSE");
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("No Warning, just echoing");

    Image4DType::Pointer input1;
    Image4DType::Pointer input2;
    Image3DType::Pointer output;

    std::vector<Activation> input;

    /* Open up the input */
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_input1() );
    reader->Update();
    input1 = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_input1().c_str());
        exit(-1);
    }
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_input2() );
    reader->Update();
    input1 = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_input2().c_str());
        exit(-1);
    }
    
    output = mse(input1, input2);
    
    //write final output
    {
    itk::ImageFileWriter<Image3DType>::Pointer out = 
        itk::ImageFileWriter<Image3DType>::New();
    out->SetInput(output);
    out->SetFileName(a_output());
    cout << "Writing: " << a_output() << endl;
    out->Update();
    }

    return 0;

}



