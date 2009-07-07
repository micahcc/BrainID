//image readers
#include "itkOrientedImage.h"
#include "itkImageFileReader.h"

//test
#include "itkImageFileWriter.h"
#include "itkImageConstIteratorWithIndex.h"

//iterators
#include "itkImageLinearConstIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"

//standard libraries
#include <ctime>
#include <cstdio>
#include <list>
#include <sstream>

#include "segment.h"

///////////////////////////////////////////////////
//testing code, write out an image for each section
int main(int argc, char** argv)
{
    // check arguments
    if(argc != 4) {
        printf("Usage: %s <4D fmri dir> <labels> <mask> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    
    Image4DType::Pointer fmri_img = read_dicom(argv[1]);

    //label index
    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
                itk::ImageFileReader<Image3DType>::New();
    labelmap_read->SetFileName( argv[2] );
    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
    labelmap_img->Update();
    
    //label index
    itk::ImageFileReader<Image3DType>::Pointer mask_read = 
                itk::ImageFileReader<Image3DType>::New();
    mask_read->SetFileName( argv[3] );
    Image3DType::Pointer mask_img = mask_read->GetOutput();
    mask_img->Update();

    std::list< SectionType > active_voxels;
    segment(fmri_img, labelmap_img, mask_img, active_voxels);


    fprintf(stderr, "Writing out every section, from active voxel list\n");
    // writer
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
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageSliceIteratorWithIndex<Image3DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(1);

    std::ostringstream os;
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    int prev_label = 0;
    while(list_it != active_voxels.end()) {
        if(prev_label != list_it->label) {
            os.str("");
            os << list_it->label << argv[4] << ".nii.gz";
            fprintf(stderr, "writing: %s\n", os.str().c_str());
            writer->SetFileName( os.str() );  
            writer->SetInput(outputImage);
            writer->Update();

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
        }

        out_index[0] = list_it->point.GetIndex()[0];
        out_index[1] = list_it->point.GetIndex()[1];
        out_index[2] = list_it->point.GetIndex()[2];
        fprintf(stdout, "%li %li %li\n", out_index[0],
                out_index[1], out_index[2]);
        out_it.SetIndex(out_index);
        out_it.Value() = list_it->point.Get();
        list_it++;
    }
}
