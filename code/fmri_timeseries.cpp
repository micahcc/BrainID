#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"

#include "segment.h"

#include <itkImageFileReader.h>

//The labelmap should already have been masked through a maxprob image for
//graymatter
int main( int argc, char **argv ) 
{
    // check arguments
    if(argc != 4) {
        printf("Usage: %s <4D fmri dir> <labels> <outfile>", argv[0]);
        return EXIT_FAILURE;
    }
    
    Image4DType::Pointer fmri_img = read_dicom(argv[1]);

    //label index
    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
                itk::ImageFileReader<Image3DType>::New();
    labelmap_read->SetFileName( argv[2] );
    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
    labelmap_img->Update();

    std::list< SectionType > active_voxels;

    int num_sections = segment(fmri_img, labelmap_img, active_voxels);
    
    Image4DType::RegionType fmri_region = fmri_img->GetRequestedRegion();

    //create a 2D output image of appropriate size.
    itk::ImageFileWriter< Image2DType >::Pointer writer = 
        itk::ImageFileWriter< Image2DType >::New();
    Image2DType::Pointer outputImage = Image2DType::New();

    Image2DType::RegionType out_region;
    Image2DType::IndexType out_index;
    Image2DType::SizeType out_size;
    out_size[0] = num_sections;
    out_size[1] = fmri_region.GetSize()[3];

    out_index[0] = 0;
    out_index[1] = fmri_region.GetIndex()[3];

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageIteratorWithIndex<Image2DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());

    std::list< SectionType >::iterator list_it = active_voxels.begin();
    out_it.GoToBegin();

    //Zero out the output image
    while(!out_it.IsAtEnd()) {
        while(!out_it.IsAtEndOfLine()) {
            out_it.Value() = 0;
            ++out_it;
        }
        out_it.NextLine();
    }

    //copy all the active voxels to the output image.
    while(true) {
        prev_label = 0;
        sum = 0;
        count = 0;
        while(list_it != active_voxels.end()) {
            if(list_it->label != prev_label) {
                if(count != 0) {
                    out_it.Get() = sum/count;
                    ++out_it;
                }
                prev_label = list_it->label;
                sum = 0;
                count = 0;
            }
            sum += list_it->point.Get();
            count++;
            ++list_it->point;
            list_it++;
        }
        list_it = active_voxels.begin();
        if(list_it->point.IsAtEnd()) {
            writer->SetFileName(argv[3]);  
            writer->SetInput(outputImage);
            writer->Update();
            return 0;
        }
    }


    return 0;
}

