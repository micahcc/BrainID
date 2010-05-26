#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include <itkExtractImageFilter.h>
#include <itkImageFileWriter.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include "types.h"
#include "tools.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstring>

#include <vul/vul_arg.h>

using namespace std;

typedef itk::OrientedImage<float, 1> Img1D;
typedef itk::OrientedImage<float, 2> Img2D;
typedef itk::OrientedImage<float, 3> Img3D;
typedef itk::OrientedImage<float, 4> Img4D;
typedef itk::OrientedImage<float, 5> Img5D;
typedef itk::OrientedImage<float, 6> Img6D;
typedef itk::OrientedImage<float, 7> Img7D;
typedef itk::OrientedImage<float, 8> Img8D;
typedef itk::OrientedImage<float, 9> Img9D;

#define INMAC(INTYPE, INNUM) \
    case INNUM: {\
    INTYPE::RegionType region; \
    INTYPE::SizeType size; \
    INTYPE::IndexType index; \
    for(unsigned int i = 0 ; i < INNUM; i++) { \
        index[i] = a_points()[i*2]; \
        size[i] = a_points()[i*2+1]; \
    } \
    region.SetSize(size); \
    region.SetIndex(index); \
    itk::ImageFileReader< INTYPE >::Pointer reader =  \
    itk::ImageFileReader< INTYPE >::New(); \
    reader->SetImageIO(itk::modNiftiImageIO::New()); \
    reader->SetFileName( a_input() );  \
    switch(outdim){  \
        OUTMAC(INTYPE, Img1D, 1); \
        OUTMAC(INTYPE, Img2D, 2); \
        OUTMAC(INTYPE, Img3D, 3); \
        OUTMAC(INTYPE, Img4D, 4); \
        OUTMAC(INTYPE, Img5D, 5); \
        OUTMAC(INTYPE, Img6D, 6); \
        OUTMAC(INTYPE, Img7D, 7); \
        OUTMAC(INTYPE, Img8D, 8); \
        OUTMAC(INTYPE, Img9D, 9); \
    } \
    } break
    

#define OUTMAC(INTYPE, OUTTYPE, OUTNUM) \
    case OUTNUM: { \
    itk::ExtractImageFilter< INTYPE, OUTTYPE >::Pointer filter = \
                itk::ExtractImageFilter< INTYPE , OUTTYPE >::New(); \
    filter->SetInput(reader->GetOutput()); \
    filter->SetExtractionRegion(region); \
    itk::ImageFileWriter< OUTTYPE >::Pointer writer = \
                itk::ImageFileWriter< OUTTYPE >::New(); \
    writer->SetInput(filter->GetOutput()); \
    writer->SetFileName(a_output()); \
    writer->Update(); \
    } break


int main (int argc, char** argv)
{
    vul_arg<string> a_input(0, "Input Image");
    vul_arg<string> a_output(0, "Output Image");
    vul_arg< vcl_vector<unsigned int> > a_points("-e", "[X Xlen] [Y Ylen] ... (up to 8 dimensions)");

    vul_arg_parse(argc,argv);

    std::cout << a_points().size() << std::endl;

    Image4DType::Pointer input;
    std::cout << a_input() << " " << a_output() << std::endl;

    int indim = a_points().size()/2;
    int outdim = 0;
    for(unsigned int i = 1 ; i < a_points().size() ; i+=2) {
        if(a_points()[i] == 0) {
            std::cout << "Collapsing: " << i/2 << std::endl;
        } else {
            outdim++;
            std::cout << "Size: " << i/2 << " " << a_points()[i] << std::endl;
        }
    }

    std::cout << "Inputdim: " << indim << std::endl;
    std::cout << "Outdim: " << outdim << std::endl;

    
    switch(indim){
        INMAC(Img1D, 1);
        INMAC(Img2D, 2);
        INMAC(Img3D, 3);
        INMAC(Img4D, 4);
        INMAC(Img5D, 5);
        INMAC(Img6D, 6);
        INMAC(Img7D, 7);
        INMAC(Img8D, 8);
        INMAC(Img9D, 9);
    }

    return 0;
}

