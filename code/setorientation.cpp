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
    if(argc != 3) {
        cout << "Usage:" << endl << argv[0] << " <img> <out>" << endl;
        exit(-1);
    }

    Image4DType::Pointer img;
    {
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( argv[1] );
    reader->Update();
    img = reader->GetOutput();
    }

    Image4DType::PointType origin = img->GetOrigin();;
    Image4DType::SpacingType spacing = img->GetSpacing();
    Image4DType::SizeType size = img->GetRequestedRegion().GetSize();
    Image4DType::DirectionType direction = img->GetDirection();
    cout << "Source: " << endl;
    printf("Origin: %f %f %f %f\n", origin[0], origin[1], origin[2], origin[3]);
    printf("Spacing: %f %f %f %f\n", spacing[0], spacing[1], spacing[2], spacing[3]);
    printf("Size: %lu %lu %lu %lu\n", size[0], size[1], size[2], size[3]);
    printf("Real Size: %f %f %f %f\n", size[0]*spacing[0], size[1]*spacing[1], 
                size[2]*spacing[2], size[3]*spacing[3]);
    printf("Direction: \n");
    for(int i = 0 ; i < 3 ; i++) {
        for(int j = 0 ; j < 3 ; j++) {
            printf("%f ",  direction(j,i));
        }
        printf("\n");
    }

    printf("New Origin? (4 numbers) ctr-d to skip ");
    cin >> origin[0] >> origin[1] >> origin[2] >> origin[3];
    if(!cin.eof()) 
        img->SetOrigin(origin);
    cin.clear();
    
    printf("New Spacing? (4 numbers) ctr-d to skip ");
    cin >> spacing[0] >> spacing[1] >> spacing[2] >> spacing[3];
    if(!cin.eof()) 
        img->SetSpacing(spacing);
    cin.clear();
    
    printf("New Direction? (16 numbers) ctr-d to skip ");
    for(int i = 0 ; i < 4 ; i++) {
        for(int j = 0 ; j < 4 ; j++) {
            cin >> direction(j,i);
        }
    }
    if(!cin.eof()) 
        img->SetDirection(direction);
    cin.clear();
    
    printf("Writing\n");
    ImageWriterType::Pointer writer = ImageWriterType::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName( argv[2] );
    writer->SetInput( img );
    writer->Update();

    return 0;
}

