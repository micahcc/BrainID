#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"

#include "segment.h"

#include <itkImageFileReader.h>

typedef itk::OrientedImage<double, 2> Image2DType;

//The labelmap should already have been masked through a maxprob image for
//graymatter
int main( int argc, char **argv ) 
{
    // check arguments
    if(argc != 4) {
        printf("Usage: %s <4D fmri dir> <labels> <outfile>\n", argv[0]);
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
   
    fprintf(stderr, "Segmentation complete\n");
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
    
    fprintf(stderr, "Output will be %lu x %lu\n", out_size[0], out_size[1]);

    out_index[0] = 0;
    out_index[1] = fmri_region.GetIndex()[3];
    
    fprintf(stderr, "Region will be %lu x %lu\n", out_index[0], out_index[1]);

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageLinearIteratorWithIndex<Image2DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetDirection(0);

    std::list< SectionType >::iterator list_it = active_voxels.begin();
    out_it.GoToBegin();

    fprintf(stderr, "Zero-Filling Output\n");
    //Zero out the output image
    while(!out_it.IsAtEnd()) {
        while(!out_it.IsAtEndOfLine()) {
            out_it.Value() = 0;
            ++out_it;
        }
        out_it.NextLine();
    }
    
//    list_it = active_voxels.begin();
//    fprintf(stderr, "Debugging...\n");
//    while(list_it != active_voxels.end()) {
//        fprintf(stderr, "label: %d, prev_label: %i\n", list_it->label, prev_label);
//        if(list_it->label != prev_label) {
//            prev_label = list_it->label;
//            printf("%d\n", prev_label);
//        }
//        list_it++;
//    }

    fprintf(stderr, "Summing GM voxels\n");
    out_it.GoToBegin();
    int prev_label = -1;
    double sum;
    int count = 0;
    while(!out_it.IsAtEnd()) {
        list_it = active_voxels.begin();
        while(!out_it.IsAtEndOfLine()) {
            prev_label = list_it->label;
            sum = 0;
            count = 0;
            while(list_it->label == prev_label) {
                sum += list_it->point.Get();
                ++(list_it->point);
                count++;
            }
            fprintf(stderr, "Finished summing section: %i\n", prev_label);
            fprintf(stderr, "Placing %f at %lu, %lu\n", out_it.Get(), 
                    out_it.GetIndex()[0], out_it.GetIndex()[1]);
            out_it.Value() = (sum/count);
            ++out_it;
        }
        out_it.NextLine();
    }

//    //copy all the active voxels to the output image.
//    while(!list_it->point.IsAtEnd()) {
//        prev_label = -1;
//        list_it = active_voxels.begin();
//        while(list_it != active_voxels.end()) {
////            fprintf(stderr, "label: %d, prev_label: %i\n", list_it->label, prev_label);
//            if(list_it->label != prev_label) {
//                if(prev_label != -1) {
//                    fprintf(stderr, "Finished summing section: %i\n", prev_label);
//                    out_it.Set(sum/count);
//                    fprintf(stderr, "Placing %f at %lu, %lu\n", out_it.Get(), 
//                                out_it.GetIndex()[0], out_it.GetIndex()[1]);
//                    ++out_it;
//                }
//                sum = 0;
//                count = 0;
//                prev_label = list_it->label;
//            } 
//            sum += list_it->point.Get();
//            count++;
//            ++(list_it->point);
//            list_it++;
//        }
//        out_it.NextLine();
//    }

    fprintf(stderr, "Writing Image\n");
    writer->SetFileName(argv[3]);  
    writer->SetInput(outputImage);
    writer->Update();

    return 0;
}

