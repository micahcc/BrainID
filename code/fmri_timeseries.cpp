#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"

#include <itkImageFileReader.h>

#include <sstream>

typedef itk::OrientedImage<double, 4> ImageTimeSeries;

ImageTimeSeries::Pointer init4DImage(ImageTimeSeries::SizeType size, size_t section,
            itk::MetaDataDictionary dic)
{
    //create a 1D output image of appropriate size.
    ImageTimeSeries::Pointer outputImage = ImageTimeSeries::New();

    ImageTimeSeries::RegionType out_region;
    ImageTimeSeries::IndexType out_index = {{0,0,0,0}};
    
    fprintf(stderr, "Region will be %lu x %lu\n", out_index[0], out_index[1]);

    out_region.SetSize(size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    itk::EncapsulateMetaData<size_t>(dic, "section", section);
    outputImage->SetMetaDataDictionary( dic );
    outputImage->Allocate();
    return outputImage;
}

ImageTimeSeries::Pointer initTimeSeries(size_t tlen, size_t section, size_t slice,
            itk::MetaDataDictionary dic)
{
    //create a 1D output image of appropriate size.
    ImageTimeSeries::Pointer outputImage = ImageTimeSeries::New();

    ImageTimeSeries::RegionType out_region;
    ImageTimeSeries::IndexType out_index = {{0,0,0,0}};
    ImageTimeSeries::SizeType out_size;
    
    out_size[0] = 1;
    out_size[1] = 1;
    out_size[2] = 1;
    out_size[3] = tlen;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    itk::EncapsulateMetaData<size_t>(dic, "section", section);
    itk::EncapsulateMetaData<size_t>(dic, "slice", slice);
    outputImage->SetMetaDataDictionary( dic );
    outputImage->Allocate();
    return outputImage;
}

void resetVoxels(std::list< SectionType>& active_voxels){
    //put all the voxels at the beginning of time
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    while(list_it != active_voxels.end()) {
        list_it->point.GoToBegin();
        list_it++;
    }
}


double get_average(std::list< SectionType >& active_voxels)
{
    double sum = 0;
    int count = 0;
    fprintf(stderr, "Summing GM voxels\n");
    resetVoxels(active_voxels);

    std::list< SectionType >::iterator list_it = active_voxels.begin();
    while(!active_voxels.begin()->point.IsAtEnd()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through it steps time for each voxel
        list_it = active_voxels.begin();
        while(list_it != active_voxels.end()) {
                sum += list_it->point.Get();
                ++(list_it->point); //move forward in time
                count++;
                ++list_it;
        }
    }

    //calculate average levels from sum
    fprintf(stderr, "Number of voxels-time: %i\n", count);
    fprintf(stderr, "Average Voxel Level: %f\n", sum/count);
    return sum/count;
}

void writeSections4D(std::list< SectionType >& active_voxels, std::string prefix,
            double average, itk::MetaDataDictionary& dic)
{
    int label;
    size_t image_i;
    std::ostringstream oss;
    
    std::vector< ImageTimeSeries::Pointer > imageout_list;
    
    itk::ImageFileWriter< ImageTimeSeries >::Pointer writer = 
        itk::ImageFileWriter< ImageTimeSeries >::New();
    
    //put all the time iterators back to 0
    resetVoxels(active_voxels);
   
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    //move through active voxels and copy them to the image with
    //the matching label
    while(!active_voxels.begin()->point.IsAtEnd()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through it steps time for each voxel
        list_it = active_voxels.begin();
        image_i = 0;
        label = list_it->label; //get first label
        while(list_it != active_voxels.end()) {

            //since active_voxels is sorted by label/slice, once the label
            //  changes it is time to move to the next section
            if(list_it->label != label) {
                image_i++;
                label = list_it->label;
            }

            //if image for this slice/section does not exist add it
            if(image_i+1 > imageout_list.size()) {
                fprintf(stderr, "Creating new region for section %i\n", label);
                imageout_list.push_back(
                            init4DImage(active_voxels.begin()->point.GetRegion().GetSize(),
                            label, dic) ); //section, dictionary
            }

            //set the current voxel in the output image to the current voxel in the
            //input image, after normalizing
            imageout_list[image_i]->SetPixel(list_it->point.GetIndex(), 
                        (list_it->point.Get()-average)/average);
            
            ++(list_it->point); //move forward in time
            ++list_it;
        }
    }
    
    //write out the images
    for(unsigned int i = 0 ; i<imageout_list.size() ; i++) {
    //setup iterators
        oss.str("");
        itk::ExposeMetaData(imageout_list[i]->GetMetaDataDictionary(), 
                    "section", label);
        oss << prefix << "_" << label << ".nii.gz";
        writer->SetFileName(oss.str());
        writer->SetInput(imageout_list[i]);
        fprintf(stderr, "Writing Image %s\n", oss.str().c_str());
        writer->Update();
    }
}

void writeSectionsTimeseries(std::list< SectionType >& active_voxels, std::string prefix,
            double average, itk::MetaDataDictionary& dict)
{
    int label;
    int slice;
    double sum;
    size_t count;
    size_t image_i;
    std::ostringstream oss;
    
    std::vector< ImageTimeSeries::Pointer > timeseries_out;
    
    itk::ImageFileWriter< ImageTimeSeries >::Pointer writer = 
        itk::ImageFileWriter< ImageTimeSeries >::New();
    
    //put all the time iterators back to 0
    resetVoxels(active_voxels);
    
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    //sum up each section for each time period
    //while none of the active_voxel iterators have reached the end
    while(!active_voxels.begin()->point.IsAtEnd()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through it steps time for each voxel
        list_it = active_voxels.begin();
        image_i = 0;
        while(list_it != active_voxels.end()) {
            label = list_it->label;
            slice = list_it->point.GetIndex()[3];
            sum = 0;
            count = 0;

            //since active_voxels is sorted by label/slice, once the label or slice
            //  changes it is time to move to the next section/slice
            while(list_it->label == label && 
                        list_it->point.GetIndex()[2] == slice) {
                sum += list_it->point.Get();
                ++(list_it->point); //move forward in time
                count++;
                ++list_it;
            }

            fprintf(stderr, "Finished summing section: %i", label);
            fprintf(stderr, " Slice : %i", slice);

            //if image for this slice/section does not exist add it
            if(image_i+1 > timeseries_out.size()) {
                fprintf(stderr, "Creating new region afformentioned section/slice\n");
                timeseries_out.push_back( initTimeSeries(
                            active_voxels.begin()->point.GetRegion().GetSize()[3], //timelen
                            label, slice, dict) ); //section, slice
            }

            //set the voxel in the timeseries equal to the normalized average
            //for the section/slice at the current time
            ImageTimeSeries::IndexType index = {{ 0,0,0, list_it->point.GetIndex()[3] }};
            timeseries_out[image_i]->SetPixel(index, (sum/count-average)/average);
            
            //move forward in timeseries_out
            image_i++;
        }
    }
    
    //write out images
    for(unsigned int i = 0 ; i<timeseries_out.size() ; i++) {
    //setup iterators
        oss.str("");
        itk::ExposeMetaData(timeseries_out[i]->GetMetaDataDictionary(), 
                    "slice", slice);
        itk::ExposeMetaData(timeseries_out[i]->GetMetaDataDictionary(), 
                    "section", label);
        oss << prefix << "_" << label << "_" << slice << ".nii.gz";
        writer->SetFileName(oss.str());
        writer->SetInput(timeseries_out[i]);
        fprintf(stderr, "Writing Image %s\n", oss.str().c_str());
        writer->Update();
    }
}

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    // check arguments
    if(argc != 5) {
        printf("Usage: %s <4D fmri dir> <labels> <4D-output> <timeseries prefix>", argv[0]);
        printf("\t<prefix for region images>");
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

    fprintf(stderr, "Grabbing Segments\n");
    int num_sections = segment(fmri_img, labelmap_img, active_voxels);
    fprintf(stderr, "Done with %i active sections.\n", num_sections);

    fprintf(stderr, "Averaging over time and space...\n");
    double average = get_average(active_voxels);
    fprintf(stderr, "Done with average: %f\n", average);

    fprintf(stderr, "Generating Time images\n");
    writeSectionsTimeseries(active_voxels, argv[4], average, 
                fmri_img->GetMetaDataDictionary());
    fprintf(stderr, "Done.\n");
    
    fprintf(stderr, "Generating 4D FMRI image for each section.\n");
    writeSections4D(active_voxels, argv[5], average, 
                fmri_img->GetMetaDataDictionary());
    fprintf(stderr, "Done.\n");

    return 0;
}

