#include "segment.h"

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

    std::list< SectionType* > active_voxels;

    sort_voxels(fmri_img, labelmap_img, active_voxels);
   


    return 0;
}

