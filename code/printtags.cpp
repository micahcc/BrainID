#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include <string>
#include <iostream>
#include <vector>

using namespace std;

typedef float ImagePixelType;
typedef itk::OrientedImage< ImagePixelType,  4 > Image4DType;
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
        double tmpd = 0;
        float tmpf = 0;
        string tmps = 0;
        int tmpi = 0;
        unsigned int tmpu = 0;
        long tmpl = 0;
        size_t tmpz = 0;
        if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpd)) {
            itk::ExposeMetaData(dict, keys[i], tmpd);
            cout << keys[i] << " -> " << tmpd << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpf)) {
            itk::ExposeMetaData(dict, keys[i], tmpf);
            cout << keys[i] << " -> " << tmpf << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmps)) {
            itk::ExposeMetaData(dict, keys[i], tmps);
            cout << keys[i] << " -> " << tmps << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpi)) {
            itk::ExposeMetaData(dict, keys[i], tmpi);
            cout << keys[i] << " -> " << tmpi << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpu)) {
            itk::ExposeMetaData(dict, keys[i], tmpu);
            cout << keys[i] << " -> " << tmpu << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpz)) {
            itk::ExposeMetaData(dict, keys[i], tmpz);
            cout << keys[i] << " -> " << tmpz << endl;
        } else if(dict[keys[i]]->GetMetaDataObjectTypeInfo() == typeid(tmpl)) {
            itk::ExposeMetaData(dict, keys[i], tmpl);
            cout << keys[i] << " -> " << tmpl << endl;
        } else {
            cout << "Type unhandled:" << dict[keys[i]]->GetMetaDataObjectTypeName()
                        << endl;
        }
    }

    fprintf(stderr, "Size: \n");
    fprintf(stderr, "%lu,", image->GetRequestedRegion().GetSize()[0]);
    fprintf(stderr, "%lu,", image->GetRequestedRegion().GetSize()[1]);
    fprintf(stderr, "%lu,", image->GetRequestedRegion().GetSize()[2]);
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Index: \n");
    fprintf(stderr, "%li,", image->GetRequestedRegion().GetIndex()[0]);
    fprintf(stderr, "%li,", image->GetRequestedRegion().GetIndex()[1]);
    fprintf(stderr, "%li,", image->GetRequestedRegion().GetIndex()[2]);
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Spacing: \n");
    fprintf(stderr, "%f ", image->GetSpacing()[0]);
    fprintf(stderr, "%f ", image->GetSpacing()[1]);
    fprintf(stderr, "%f ", image->GetSpacing()[2]);
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Orientation\n");
    for(int ii = 0 ; ii < 3 ; ii++) {
        for(int jj = 0 ; jj < 3 ; jj++) {
            fprintf(stderr, "%f ", image->GetDirection()(jj,ii));
        }
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Origin\n");
    for(int ii = 0 ; ii < 3 ; ii++)  {
        fprintf(stderr, "%f ", image->GetOrigin()[ii]);
    }
    fprintf(stderr, "\n");

    return 0;
}


