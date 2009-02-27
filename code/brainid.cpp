#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkVector.h>
#include "bold.h"

#include <cstdio> 
#include <cstdlib>
#include <cstring>

using namespace std;

//Load observations
typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

int main(int argc, char** argv)
{
    
    if(argc != 3) {
        printf("Usage: %s <inputname> <outputname>", argv[0]);
    }
    
    long lNumber = 1000;
    long lIterates;

    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    //Initialise and run the sampler

      double xm,xv,ym,yv;
      
      cout << xm << "," << ym << "," << xv << "," << yv << endl;

}
