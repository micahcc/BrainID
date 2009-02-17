//image readers
#include <itkOrientedImage.h>
#include <itkImageFileReader.h>

//test
#include <itkImageFileWriter.h>
#include <itkImageConstIteratorWithIndex.h>

//iterators
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>

//standard libraries
#include <cstdio>
#include <list>
#include <sstream>

#include "segment.h"

////////////////////////////////////////////////////
//Testing by writing out each timestep as a 3D image
int main(int argc, char** argv)
{
    // check arguments
    if(argc != 4) {
        printf("Usage: %s <4D fmri dir> <labels> <outfile>", argv[0]);
        return EXIT_FAILURE;
    }
    
    Image4DType::Pointer fmri_img = read_dicom(argv[1]);

    //perform test
    fprintf(stderr, "Showing every time in fmri image\n");
    Image4DType::RegionType fmri_region = fmri_img->GetRequestedRegion();

    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();

    Image3DType::Pointer outputImage = Image3DType::New();

    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;
    out_size[0] = fmri_region.GetSize()[0];
    out_size[1] = fmri_region.GetSize()[1];
    out_size[2] = fmri_region.GetSize()[2];

    out_index[0] = fmri_region.GetIndex()[0];
    out_index[1] = fmri_region.GetIndex()[1];
    out_index[2] = fmri_region.GetIndex()[2];

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
//    outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageSliceIteratorWithIndex<Image3DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(1);

    std::ostringstream os;
    
    SliceIterator4D fmri_it( fmri_img, fmri_img->GetRequestedRegion() );
  
    fmri_it.SetFirstDirection(0);
    fmri_it.SetSecondDirection(1);
    fmri_it.GoToBegin();
    
    PixelIterator4D time_it( fmri_img, fmri_img->GetRequestedRegion() );
    time_it.SetDirection(3);
    time_it.GoToBegin();
    ++time_it;
    int count = 0;
    do {
        //zero output
        out_it.GoToBegin();
        while(!out_it.IsAtEnd()) {
            fprintf(stderr, ".");
            while(!out_it.IsAtEndOfSlice()) {
                while(!out_it.IsAtEndOfLine()) {
                    out_it.Value() = 0;
                    ++out_it;
                }
                out_it.NextLine();
            }
            out_it.NextSlice();
        }

        //write out time slice
        while(fmri_it != time_it && !fmri_it.IsAtEnd()) {
            while(!fmri_it.IsAtEndOfSlice()) {
                while(!fmri_it.IsAtEndOfLine()) {
                    out_it.Value() = fmri_it.Get();
                    ++fmri_it;
                    ++out_it;
                }
                fmri_it.NextLine();
                out_it.NextLine();
            }
            fmri_it.NextSlice();
            out_it.NextSlice();
        }

        os.str("");
        os << count++ << argv[3];
        fprintf(stderr, "%s\n", os.str().c_str());
        writer->SetFileName( os.str() );  
        writer->SetInput(outputImage);
        writer->Update();
        ++time_it;
    } while(!time_it.IsAtEnd() && !fmri_it.IsAtEnd());

//    fprintf(stderr, "%li %li %li %li and %li %li %li %li\n", fmri_it.GetIndex()[0],
//            fmri_it.GetIndex()[1], fmri_it.GetIndex()[2], fmri_it.GetIndex()[3], 
//            time_it.GetIndex()[0], time_it.GetIndex()[1], time_it.GetIndex()[2],
//            time_it.GetIndex()[3]);
}
