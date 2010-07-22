#include "itkOrientedImage.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

#include <iostream>
#include "tools.h"

using namespace std;

int main (int argc, char** argv)
{
    if(argc != 4) {
        cout << "Usage:" << endl << argv[0] << " <src> <dest> <out>" << endl;
        cout << "For each voxel in dest, the nearest voxel is found in source" << endl;
        return 0;
    }

    Image4DType::Pointer input;
    {
    itk::ImageFileReader< Image4DType >::Pointer reader = 
                itk::ImageFileReader< Image4DType >::New();
    reader->SetFileName( argv[1] );
    reader->Update();
    input = reader->GetOutput();
    }

    Image4DType::Pointer dest;
    {
    itk::ImageFileReader< Image4DType >::Pointer reader = 
                itk::ImageFileReader< Image4DType >::New();
    reader->SetFileName( argv[2] );
    reader->Update();
    dest = reader->GetOutput();
    }

    try {
        itk::ImageFileReader< Image4DType >::Pointer reader = 
                    itk::ImageFileReader< Image4DType >::New();
        reader->SetFileName( argv[3] );
        reader->Update();
        cout << "Hmm " << argv[3] << " already exists, please move or delete" << endl;
        return -1;
    } catch (...) { }

    Image4DType::Pointer out = Image4DType::New();
    out->SetRegions(dest->GetRequestedRegion().GetSize());
    out->Allocate();
    out->FillBuffer(0);
    out->CopyInformation(dest);
    
    Image4DType::IndexType index;
    Image4DType::PointType point;
    itk::ImageRegionIterator<Image4DType> it(out, out->GetRequestedRegion());
    while(!it.IsAtEnd()) {
        out->TransformIndexToPhysicalPoint(it.GetIndex(), point);
        input->TransformPhysicalPointToIndex(point, index);
        if(input->GetRequestedRegion().IsInside(index))
            it.Set(input->GetPixel(index));

        ++it;
    }
    
    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetFileName( argv[3] );
    writer->SetInput(out);
    writer->Update();

    return 0;
}

