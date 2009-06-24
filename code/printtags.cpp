#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include <string>
#include <iostream>
#include <vector>

using namespace std;

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;

int main (int argc, char** argv)
{
    if(argc != 2) {
        cout << "Usage:" << endl << argv[0] << " <filename>" << endl;
    }
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( argv[1] );
    reader->Update();

    Image4DType::Pointer image = reader->GetOutput();

    itk::MetaDataDictionary dict = reader->GetOutput()->GetMetaDataDictionary();
    vector<string> keys = dict.GetKeys();

//    dict.Print(cout);

//    cout << "Manually Printing" << endl;

    for(size_t i=0 ; i<keys.size() ; i++) {
        double tmpd;
        float tmpf;
        string tmps;
        int tmpi;
        unsigned int tmpu;
        size_t tmpz;
        if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpd)) {
            itk::ExposeMetaData<double>(dict, keys[i], tmpd);
            cout << keys[i] << " -> " << tmpd << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpf)) {
            itk::ExposeMetaData<float>(dict, keys[i], tmpf);
            cout << keys[i] << " -> " << tmpf << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmps)) {
            itk::ExposeMetaData<string>(dict, keys[i], tmps);
            cout << keys[i] << " -> " << tmps << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpi)) {
            itk::ExposeMetaData<int>(dict, keys[i], tmpi);
            cout << keys[i] << " -> " << tmpi << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpu)) {
            itk::ExposeMetaData<unsigned int>(dict, keys[i], tmpu);
            cout << keys[i] << " -> " << tmpu << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpz)) {
            itk::ExposeMetaData<size_t>(dict, keys[i], tmpz);
            cout << keys[i] << " -> " << tmpz << endl;
        } else {
            cout << "Other" << endl;
        }
    }
    
    return 0;
}


