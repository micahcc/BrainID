#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include "tools.h"

#include <string>
#include <iostream>
#include <vector>

#include <vul/vul_arg.h>

using namespace std;

typedef itk::OrientedImage< double,  4 > Image4DType;
typedef itk::OrientedImage< int,  3 > Image3DType;

int main (int argc, char** argv)
{
    if(argc != 4) {
        cout << "Usage:" << endl << argv[0] << " <src> <mask> <out>" << endl;
        return 0;
    }

    Image4DType::Pointer input;
    {
    itk::ImageFileReader< Image4DType >::Pointer reader = 
                itk::ImageFileReader< Image4DType >::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( argv[1] );
    reader->Update();
    input = reader->GetOutput();
    }

    Image3DType::Pointer mask;
    {
    itk::ImageFileReader< Image3DType >::Pointer reader = 
                itk::ImageFileReader< Image3DType >::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( argv[2] );
    reader->Update();
    mask = reader->GetOutput();
    }
    

    Image4DType::Pointer out = applymask<double, 4, int, 3>(input, mask);

    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName( argv[3] );
    writer->SetInput(out);
    writer->Update();

    return 0;
}
