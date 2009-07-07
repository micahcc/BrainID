#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include <string>
#include <iostream>
#include <vector>

using namespace std;

typedef float ImagePixelType;
typedef itk::OrientedImage< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  ImageWriterType;

int main (int argc, char** argv)
{
    if(argc != 2) {
        cout << "Usage:" << endl << argv[0] << " <src> <dst> <out>" << endl;
    }
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( argv[1] );
    reader->Update();

    Image4DType::Pointer image = reader->GetOutput();
    
    ImageReaderType::Pointer reader2 = ImageReaderType::New();
    reader2->SetImageIO(itk::modNiftiImageIO::New());
    reader2->SetFileName( argv[2] );
    reader2->Update();

    Image4DType::Pointer image2 = reader2->GetOutput();

    image2->SetOrigin(image->GetOrigin());

    ImageWriterType::Pointer writer = ImageWriterType::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName( argv[3] );
    writer->SetInput( image2 );
    writer->Update();

    return 0;
}
