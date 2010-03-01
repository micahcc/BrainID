#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkMetaDataObject.h"
#include "modNiftiImageIO.h"

#include <cstdio>

using namespace std;

typedef itk::OrientedImage<unsigned short, 4> Image4DType;

Image4DType::Pointer init4DImage(size_t xlen, size_t ylen, 
            size_t zlen, size_t tlen)
{
    Image4DType::Pointer out = Image4DType::New();
    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index;
    Image4DType::SizeType out_size;

    out_size[0] = xlen;
    out_size[1] = ylen;
    out_size[2] = zlen;
    out_size[3] = tlen; 
    
    out_index[0] = 0;
    out_index[1] = 0;
    out_index[2] = 0;
    out_index[3] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    itk::EncapsulateMetaData<string>(out->GetMetaDataDictionary(), "VERSION", 
                BRAINID_VERSION);
    
    out->SetRegions( out_region );
    out->Allocate();
    return out;
}


int main (int argc, char** argv)
{
    Image4DType::Pointer newimage = init4DImage(9,5,7,3);

    for(int xx = 0 ; xx < 9; xx++) {
        for(int yy = 0 ; yy < 5; yy++) {
            for(int zz = 0 ; zz < 7; zz++) {
                for(int tt = 0 ; tt < 3; tt++) {
                    Image4DType::IndexType index = {{xx,yy,zz,tt}};
                    unsigned short num = xx+10*yy + zz*100+tt*1000;
                    newimage->SetPixel(index, num);
                }
            }
        }
    }

    itk::ImageFileWriter<Image4DType>::Pointer write = itk::ImageFileWriter<Image4DType>::New();
    write->SetInput(newimage);
    write->SetFileName(argv[1]);
    write->Update();
    
    return 0;
}


