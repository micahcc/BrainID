#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

typedef float ImagePixelType;
typedef itk::OrientedImage< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  ImageWriterType;

int main (int argc, char** argv)
{
    if(argc != 3) {
        cout << "Usage:" << endl << argv[0] << " <img> <out>" << endl;
    }

    Image4DType::Pointer img;
    {
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( argv[1] );
    reader->Update();
    img = reader->GetOutput();
    }

    Image4DType::OriginType origin;
    Image4DType::SpacingType spacing;
    cout << "Source: " << endl;
    printf("Origin: %f %f %f\n", origin[0], origin[1], origin[2], origin[3]);
    printf("Spacing: %f %f %f\n", spacing[0], spacing[1], spacing[2], spacing[3]);

    printf("New Origin? (4 numbers) ");
    cin >> origin[0] >> origin[1] >> origin[2] >> origin[3];

    img->SetOrigin(origin);
    ImageWriterType::Pointer writer = ImageWriterType::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName( argv[2] );
    writer->SetInput( img );
    writer->Update();

    return 0;
}
