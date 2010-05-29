#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "itkMetaDataObject.h"

#include <iomanip>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

typedef float PixelType;

template <unsigned int DIM>
class ImageType : public itk::OrientedImage< PixelType, DIM> {};

template <unsigned int DIM>
void changeOrient(string name, string out)
{
    typedef ImageType<DIM> Image;
    typedef itk::ImageFileReader< itk::OrientedImage<PixelType, DIM> > Reader;
    typedef itk::ImageFileWriter< itk::OrientedImage<PixelType, DIM> > Writer;

    typename Reader::Pointer reader = Reader::New();
    reader->SetFileName(name);
    reader->Update();

    typename Image::PointType origin = reader->GetOutput()->GetOrigin();;
    typename Image::SpacingType spacing = reader->GetOutput()->GetSpacing();
    typename Image::SizeType size = reader->GetOutput()->GetRequestedRegion().GetSize();
    typename Image::DirectionType direction = reader->GetOutput()->GetDirection();

    unsigned int dims = reader->GetOutput()->GetImageDimension();

    cerr << "Origin: " << endl;
    for(unsigned int ii = 0 ; ii < dims ; ii++)
        cerr << setw(12) << reader->GetOutput()->GetOrigin()[ii];
    cerr << endl << endl;
    
    cerr << "New Origin?" << endl;
    for(unsigned int ii = 0 ; ii < dims ; ii++)
        cin >> origin[ii];
    
    if(!cin.eof()) 
        reader->GetOutput()->SetOrigin(origin);
    cin.clear();

    cerr << "Spacing:" << endl;
    for(unsigned int ii = 0 ; ii < dims ; ii++)
        cerr << setw(12) << reader->GetOutput()->GetSpacing()[ii];
    cerr << endl << endl;
    cerr << "New Spacing?" << endl;
    for(unsigned int ii = 0 ; ii < dims ; ii++)
        cin >> spacing[ii];
    
    if(!cin.eof()) 
        reader->GetOutput()->SetSpacing(spacing);
    cin.clear();

    cerr << "Direction:" << endl;
    for(unsigned int ii = 0 ; ii < dims ; ii++) {
        for(unsigned int jj = 0 ; jj < dims ; jj++) {
            cerr << setw(12) << reader->GetOutput()->GetDirection()(jj,ii);
        }
        cerr << "\n";
    }
    cerr << endl << endl;
    cerr << "New Direction?" << endl;
    for(unsigned int ii = 0 ; ii < dims ; ii++) {
        for(unsigned int jj = 0 ; jj < dims ; jj++) {
            cin >> direction(jj,ii);
        }
    }
    if(!cin.eof()) 
        reader->GetOutput()->SetDirection(direction);
    cin.clear();
    
    cerr << "Writing" << endl;
    typename Writer::Pointer writer = Writer::New();
    writer->SetFileName(out);
    writer->SetInput( reader->GetOutput() );
    writer->Update();
}

int main (int argc, char** argv)
{
    if(argc != 3) {
        cerr << "Usage:" << endl << argv[0] << " <img> <out>" << endl;
        exit(-1);
    }

    unsigned int dims = 0;
        
    itk::ImageIOBase::Pointer io = itk::ImageIOFactory::CreateImageIO(
            argv[1], itk::ImageIOFactory::ReadMode);
    io->SetFileName(argv[1]);
    io->ReadImageInformation();
    dims = io->GetNumberOfDimensions();
    
    cerr << "Done " << endl;
    switch(dims) {
        case 2: {
            changeOrient<2>(argv[1], argv[2]);
        } break;
        case 3: {
            changeOrient<3>(argv[1], argv[2]);
        } break;
        case 4: {
            changeOrient<4>(argv[1], argv[2]);
        } break;
    }


    return 0;
}

