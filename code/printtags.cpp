#include "itkOrientedImage.h"
#include "itkImageIOFactory.h"
#include <itkImageFileReader.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include <iomanip>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

void printInfo(itk::ImageIOBase* io, char* filename)
{
    cout << filename << endl;
    io->SetFileName(filename);
    io->ReadImageInformation();
    cout << "Image Dimensions: " << io->GetNumberOfDimensions() << endl;
    itk::MetaDataDictionary dict = io->GetMetaDataDictionary();
    vector<string> keys = dict.GetKeys();

    for(size_t i=0 ; i<keys.size() ; i++) {
        double tmpd = 0;
        float tmpf = 0;
        string tmps = "";
        int tmpi = 0;
        unsigned int tmpu = 0;
        long tmpl = 0;
        size_t tmpz = 0;
        if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpd)) {
            itk::ExposeMetaData(dict, keys[i], tmpd);
            cout << keys[i] << " d -> " << tmpd << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpf)) {
            itk::ExposeMetaData(dict, keys[i], tmpf);
            cout << keys[i] << " f -> " << tmpf << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmps)) {
            itk::ExposeMetaData(dict, keys[i], tmps);
            cout << keys[i] << " s -> " << tmps << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpi)) {
            itk::ExposeMetaData(dict, keys[i], tmpi);
            cout << keys[i] << " i -> " << tmpi << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpu)) {
            itk::ExposeMetaData(dict, keys[i], tmpu);
            cout << keys[i] << " u -> " << tmpu << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpz)) {
            itk::ExposeMetaData(dict, keys[i], tmpz);
            cout << keys[i] << " z -> " << tmpz << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpl)) {
            itk::ExposeMetaData(dict, keys[i], tmpl);
            cout << keys[i] << " l -> " << tmpl << endl;
        } else {
            cout << "Type unhandled:" << dict[keys[i]]->GetMetaDataObjectTypeName()
                << endl;
        }
    }

    cout << endl << "Size:" << endl;
    for(unsigned int ii = 0 ; ii < io->GetNumberOfDimensions() ; ii++)
        cout << setw(12) << io->GetDimensions(ii);
    cout << endl << endl;

    cout << "Spacing:" << endl;
    for(unsigned int ii = 0 ; ii < io->GetNumberOfDimensions() ; ii++)
        cout << setw(12) << io->GetSpacing(ii);
    cout << endl << endl;

    cout << "Direction:" << endl;
    for(unsigned int ii = 0 ; ii < io->GetNumberOfDimensions() ; ii++) {
        for(unsigned int jj = 0 ; jj < io->GetNumberOfDimensions() ; jj++) {
            cout << setw(12) << io->GetDirection(jj)[ii];
        }
        cout << "\n";
    }
    cout << endl << endl;

    cout << "Origin: " << endl;
    for(unsigned int ii = 0 ; ii < io->GetNumberOfDimensions() ; ii++)
        cout << setw(12) << io->GetOrigin(ii);
    cout << endl << endl;
}

int main (int argc, char** argv)
{
    itk::modNiftiImageIO::Pointer io = itk::modNiftiImageIO::New();
    for(int ii = 1 ; ii < argc ; ii++) {
        cout << argv[ii] << endl;
        if(io->CanReadFile(argv[ii])) {
            printInfo(io, argv[ii]);
        } else {
            printInfo(itk::ImageIOFactory::CreateImageIO(argv[ii], 
                        itk::ImageIOFactory::ReadMode), argv[ii]);
        }
    }

    return 0;
}


