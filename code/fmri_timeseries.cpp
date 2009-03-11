#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"

#include "segment.h"

#include <itkImageFileReader.h>

typedef itk::OrientedImage<double, 2> Image2DType;

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    int prev_label = -1;
    double sum;
    int count;
    // check arguments
    if(argc != 5) {
        printf("Usage: %s <4D fmri dir> <labels> <outfile 4D> <outfile Timeseries>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    Image4DType::Pointer fmri_img = read_dicom(argv[1]);
    
    //create a 4D image writer to save the image of appropriate size.
    itk::ImageFileWriter< Image4DType >::Pointer writer4d = 
        itk::ImageFileWriter< Image4DType >::New();
    writer4d->SetFileName(argv[3]);  
    writer4d->SetInput(fmri_img);
    writer4d->Update();

    //label index
    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
                itk::ImageFileReader<Image3DType>::New();
    labelmap_read->SetFileName( argv[2] );
    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
    labelmap_img->Update();

    std::list< SectionType > active_voxels;

    int num_sections = segment(fmri_img, labelmap_img, active_voxels);
    
    //setup vector of averages for normalization
    std::vector<double> averages(num_sections, 0.0);
   
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
    out_size[1] = fmri_region.GetSize()[3] + 1; //extra for labels
    
    fprintf(stderr, "Output will be %lu x %lu\n", out_size[0], out_size[1]);

    out_index[0] = 0;
    out_index[1] = fmri_region.GetIndex()[3];
    
    fprintf(stderr, "Region will be %lu x %lu\n", out_index[0], out_index[1]);

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    //setup iterators
    itk::ImageLinearIteratorWithIndex<Image2DType> 
                out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetDirection(0);

    std::list< SectionType >::iterator list_it = active_voxels.begin();
    out_it.GoToBegin();

    std::vector<double>::iterator averages_it = averages.begin();

    //fill in the first element of each section with the label number
    fprintf(stderr, "Filling Labels\n");
    list_it = active_voxels.begin();
    while(!out_it.IsAtEndOfLine()) {
        out_it.Value() =  list_it->label;
        prev_label = list_it->label;
        fprintf(stderr, "Placing %f at %lu, %lu\n", out_it.Get(), 
                    out_it.GetIndex()[0], out_it.GetIndex()[1]);
        ++out_it;
        while(list_it->label == prev_label) list_it++;
    }
   out_it.NextLine();
 
    //sum up each section for each time period
    fprintf(stderr, "Summing GM voxels\n");
    //iterate through all output voxels, starting with the second line (the
    //  first contains the labels already
    while(!out_it.IsAtEnd()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through is steps time for each voxel
        list_it = active_voxels.begin();
        averages_it = averages.begin();
        while(!out_it.IsAtEndOfLine()) {
            prev_label = list_it->label;
            sum = 0;
            count = 0;
            //since active_voxels is sorted by label, once the label changes its
            //  time to move to the next voxel
            while(list_it->label == prev_label) {
                sum += list_it->point.Get();
                ++(list_it->point); //move forward in time
                count++;
                ++list_it;
            }
            fprintf(stderr, "Finished summing section: %i\n", prev_label);
            fprintf(stderr, "Placing %f at %lu, %lu\n", out_it.Get(), 
                        out_it.GetIndex()[0], out_it.GetIndex()[1]);
            out_it.Value() = (sum/count);
            ++out_it;
            
            //for normalization purposes
            *averages_it += sum/count;
            averages_it++;
        }
        out_it.NextLine();
    }

    //calculate average levels from sum
    averages_it = averages.begin();
    while(averages_it != averages.end()) {
        *averages_it = (*averages_it)/fmri_region.GetSize()[3];
        fprintf(stderr, "Average: %f\n", *averages_it);
        averages_it++;
    }

    //convert absolute values in output image to percent change values
    //this means dividing each entire timeseries by the average for
    //that timeseries
    out_it.SetDirection(1); //move in the time direction first
    out_it.GoToBegin();
    ++out_it; //skip the label
    
    averages_it = averages.begin();
    while(!out_it.IsAtEnd() && averages_it != averages.end()) {
        while(!out_it.IsAtEndOfLine()) {
            out_it.Value() = out_it.Get()/(*averages_it);
            ++out_it;
        }
        averages_it++;
        out_it.NextLine();
        ++out_it; //skip the labels
    }

    fprintf(stderr, "Writing Image\n");
    writer->SetFileName(argv[4]);  
    writer->SetInput(outputImage);
    writer->Update();

    return 0;
}

