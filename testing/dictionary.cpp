#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include "segment.h"
#include <string>

typedef float ImagePixelType;
typedef itk::OrientedImage< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  ImageWriterType;

using namespace std;

int main()
{
    Image4DType::SizeType size = {{10,10,10,10}};
    Image4DType::IndexType index = {{0,0,0,0}};
    Image4DType::RegionType region;
    region.SetSize(size);
    region.SetIndex(index);
    Image4DType::Pointer out = Image4DType::New();
    out->SetRegions(region);
    out->Allocate();
    out->FillBuffer(0);

    double       tmp1d = 12.2;
    float        tmp1f = 12e12;
    int          tmp1i = -2;
    unsigned int tmp1u = 2224;
    size_t       tmp1z = 231512;
    long long int tmp1ll = -123452;
    long long unsigned int tmp1llu = 1231204;
    unsigned short tmp1us = 65000;
    short tmp1si = -32000;
    std::string tmp1str = "hello world";

    double       tmp2d = 0;
    float        tmp2f = 0;
    int          tmp2i = 0;
    unsigned int tmp2u = 0;
    size_t       tmp2z = 0;
    long long int tmp2ll = 0;
    long long unsigned int tmp2llu = 0;
    unsigned short tmp2us = 0;
    short tmp2si = 0;
    std::string tmp2str = "";
    {
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "double", tmp1d);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "float", tmp1f);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "int", tmp1i);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "unsigned int", tmp1u);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "size_t", tmp1z);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "long long int", tmp1ll);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "long long unsigned int", tmp1llu);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "unsigned short", tmp1us);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "short", tmp1si);
        itk::EncapsulateMetaData(out->GetMetaDataDictionary(), "string", tmp1str);
        
        ImageWriterType::Pointer writer = ImageWriterType::New();
        writer->SetInput(out);
        writer->SetFileName("test.nii");
        writer->SetImageIO(itk::modNiftiImageIO::New());
        writer->Update();

    }
    
    {
        ImageReaderType::Pointer reader = ImageReaderType::New();
        reader->SetFileName("test.nii");
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->Update();
        Image4DType::Pointer in = reader->GetOutput();
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "double", tmp2d);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "float", tmp2f);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "int", tmp2i);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "unsigned int", tmp2u);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "size_t", tmp2z);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "long long int", tmp2ll);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "long long unsigned int", tmp2llu);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "unsigned short", tmp2us);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "short", tmp2si);
        itk::ExposeMetaData(in->GetMetaDataDictionary(), "string", tmp2str);
    }

    if(tmp1d   == tmp2d   ) {
        cout << "Pass Double" << endl;
    } else {
        cout << "Fail Double" << endl;
    }
    if(tmp1f   == tmp2f   ) {
        cout << "Pass Float" << endl;
    } else {
        cout << "Fail Float" << endl;
    }
    if(tmp1i   == tmp2i   ) {
        cout << "Pass Int" << endl;
    } else {
        cout << "Fail Int" << endl;
    }
    if(tmp1u   == tmp2u   ) {
        cout << "Pass Unsigned" << endl;
    } else {
        cout << "Fail Unsigned" << endl;
    }
    if(tmp1z   == tmp2z   ) {
        cout << "Pass Size" << endl;
    } else {
        cout << "Fail Size" << endl;
    }
    if(tmp1ll  == tmp2ll  ) {
        cout << "Pass Long Long" << endl;
    } else {
        cout << "Fail Long Long" << endl;
    }
    if(tmp1llu == tmp2llu) {
        cout << "Pass Long Long Unsigned" << endl;
    } else {
        cout << "Fail Long Long Unsigned" << endl;
    }
    if(tmp1us  == tmp2us  ) {
        cout << "Pass Unsigned Short" << endl;
    } else {
        cout << "Fail Unsigned Short" << endl;
    }
    if(tmp1si  == tmp2si  ) {
        cout << "Pass Short" << endl;
    } else {
        cout << "Fail Short" << endl;
    }
    if(tmp1str == tmp2str ) {
        cout << "Pass String" << endl;
    } else {
        cout << "Fail String" << endl;
    }

}
