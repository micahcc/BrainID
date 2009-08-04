#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"

#include <itkImageFileReader.h>
#include "modNiftiImageIO.h"

#include <sstream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

#define TIMEDIM 3
#define SLICEDIM 0
#define SECTIONDIM 1
typedef itk::OrientedImage<double, 4> ImageTimeSeries;

using namespace std;

ImageTimeSeries::Pointer init4DImage(ImageTimeSeries::SizeType size, int section, 
            Image4DType::Pointer im)
{
    //create a 1D output image of appropriate size.
    ImageTimeSeries::Pointer outputImage = ImageTimeSeries::New();

    ImageTimeSeries::RegionType out_region;
    ImageTimeSeries::IndexType out_index = {{0,0,0,0}};
    
    fprintf(stderr, "Region will be %zu x %zu x %zu \n", size[0], size[1], size[2]);

    out_region.SetSize(size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    outputImage->CopyInformation(im);
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "section", section);
    return outputImage;
}

ImageTimeSeries::Pointer initTimeSeries(size_t tlen, int section, int numslice)
{
    fprintf(stderr, "Creating new region\n");
    fprintf(stderr, " Section    : %i\n", section);
    //create a 1D output image of appropriate size.
    ImageTimeSeries::Pointer outputImage = ImageTimeSeries::New();

    ImageTimeSeries::RegionType out_region;
    ImageTimeSeries::IndexType out_index = {{0,0,0,0}};
    ImageTimeSeries::SizeType out_size = {{1, 1, 1, 1}};
    
    out_size[SLICEDIM] = numslice;
    out_size[TIMEDIM] = tlen;
    fprintf(stderr, " numslice   : %lu\n", out_size[SLICEDIM]);
    fprintf(stderr, " tlen       : %lu\n", out_size[TIMEDIM]);
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), 
                "section", section);
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), 
                "slices", numslice);
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "Dim3", 
                std::string("time"));
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "Dim0", 
                std::string("slices"));
    return outputImage;
}

void resetVoxels(std::list< SectionType>& active_voxels){
    //put all the voxels at the beginning of time
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    while(list_it != active_voxels.end()) {
        list_it->point.GoToBeginOfLine();
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
    while(!active_voxels.begin()->point.IsAtEndOfLine()) {
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
            double average, itk::MetaDataDictionary& dic, Image4DType::Pointer im)
{
    int label;
    std::ostringstream oss;
    
    std::list< ImageTimeSeries::Pointer > imageout_list;
    
    itk::ImageFileWriter< ImageTimeSeries >::Pointer writer = 
        itk::ImageFileWriter< ImageTimeSeries >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    
    //put all the time iterators back to 0
    resetVoxels(active_voxels);
    
    Image4DType::SpacingType space4;
    space4[0] = 1.0;
    space4[1] = 1.0;
    space4[2] = 1.0;
    
//    {
//    double tmp;
//    itk::ExposeMetaData<double>(dic, "TemporalResolution", tmp);
//    fprintf(stderr, "res: %f\n", tmp);
//    }
   
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    //move through active voxels and copy them to the image with
    //the matching label
    while(!active_voxels.begin()->point.IsAtEndOfLine()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through it steps time for each voxel
        list_it = active_voxels.begin();
        label = list_it->label; //get first label
        std::list< ImageTimeSeries::Pointer >::iterator image_it = 
                    imageout_list.begin();
        while(list_it != active_voxels.end()) {

            //since active_voxels is sorted by label/slice, once the label
            //  changes it is time to move to the next section
            if(list_it->label != label) {
                image_it++;
                label = list_it->label;
            }

            //if image for this slice/section does not exist add it
            if(image_it == imageout_list.end()) {
                fprintf(stderr, "Creating new region for section %i\n", label);
                imageout_list.push_back(
                            init4DImage(
                            active_voxels.begin()->point.GetRegion().GetSize(),
                            label, im) ); //section, dictionary
                imageout_list.back()->GetMetaDataDictionary()["TemporalResolution"] =
                            dic["TemporalResolution"];
                {
                    double tmp;
                    itk::ExposeMetaData(imageout_list.back()->GetMetaDataDictionary()
                                , "TemporalResolution", tmp);
                    space4[3] = tmp;
                    imageout_list.back()->SetSpacing(space4);
                }
                image_it = imageout_list.end();
                image_it--;

            }

            //set the current voxel in the output image to the current voxel in the
            //input image, after normalizing
            (*image_it)->SetPixel(list_it->point.GetIndex(), 
                        (list_it->point.Get()-average)/average);
            
            ++(list_it->point); //move forward in time
            ++list_it;
        }
    }
    
    //write out the images
    while(!imageout_list.empty()) {
        oss.str("");
        if(!imageout_list.back()->GetMetaDataDictionary().HasKey("section")) {
            fprintf(stderr, "Error could not find metadata dictionary key for section\n");
            exit(-5);
        }
        itk::ExposeMetaData(imageout_list.back()->GetMetaDataDictionary(), 
                    "section", label);
        oss << prefix << "_" << label << ".nii.gz";
        writer->SetFileName(oss.str());
        writer->SetInput(imageout_list.back());
        fprintf(stderr, "Writing Image %s\n", oss.str().c_str());
        writer->Update();
        imageout_list.pop_back();
    }
}

void writeTimeseries(std::list< SectionType >& active_voxels, 
            std::string filename, double average, itk::MetaDataDictionary& dict)
{
    int indexcount = 0;
    int label;
    double sum;
    size_t count;
    int numsections = 1; 
    std::ostringstream oss;
    
    //calculate the number of regions
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    label =  list_it->label;
    while(list_it != active_voxels.end()) {
        if(label != list_it->label) {
            numsections++;
            label = list_it->label;
        }
        list_it++;
    }
    
    //initialize the output image
    ImageTimeSeries::Pointer timeseries_out = ImageTimeSeries::New();
    ImageTimeSeries::RegionType out_region;
    ImageTimeSeries::IndexType out_index = {{0,0,0,0}};
    ImageTimeSeries::SizeType out_size = {{1, 1, 1, 1}};
    out_size[SECTIONDIM] = numsections;
    out_size[TIMEDIM] = active_voxels.begin()->point.GetRegion().GetSize()[TIMEDIM];
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    timeseries_out->SetRegions( out_region );
    timeseries_out->Allocate();

    //set image information
    timeseries_out->GetMetaDataDictionary()["TemporalResolution"] 
                = dict["TemporalResolution"];
    {
        Image4DType::SpacingType space4;
        space4[0] = 1.0;
        space4[1] = 1.0;
        space4[2] = 1.0;
        double tmp;
        itk::ExposeMetaData(timeseries_out->GetMetaDataDictionary()
                    , "TemporalResolution", tmp);
        space4[3] = tmp;
        timeseries_out->SetSpacing(space4);
    }
    
    //put all the time iterators back to 0
    resetVoxels(active_voxels);
       
    //save metadata regarding which indices indicate which sections
    indexcount = 0;
    list_it = active_voxels.begin();
    while(list_it != active_voxels.end()){
        oss.str("");
        oss << "index" << indexcount;
        itk::EncapsulateMetaData(timeseries_out->GetMetaDataDictionary(),
                    oss.str(), list_it->label);

        //move to next label (and fast forward through list)
        indexcount++;
        label = list_it->label;
        while(list_it->label == label)
            list_it++;
    }
    
    //sum up each section for each time period
    //while none of the active_voxel iterators have reached the end
    while(!active_voxels.begin()->point.IsAtEndOfLine()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through it steps time for each voxel
        list_it = active_voxels.begin();
        //each iteration through this loop represents a single section
        //in the active_voxels list 
        indexcount = 0;
        while(list_it != active_voxels.end()) {
            //convenience variables
            label = list_it->label;
            int time = list_it->point.GetIndex()[3];
#ifdef DEBUG
            fprintf(stderr, "Label: %i Slice: %i Time: %i\n", label, slice, time);
#endif //DEBUG

            //used to get average over region
            sum = 0;
            count = 0;

            //since active_voxels is sorted by label/slice, once the label or slice
            //  changes it is time to move to the next section/slice
            while(list_it->label == label) {
                sum += list_it->point.Get();
                ++(list_it->point); //move forward in time
                count++;
                list_it++;
            }

            //set the voxel in the timeseries equal to the normalized average
            //for the section/slice at the current time
            ImageTimeSeries::IndexType index = {{ 0,0,0,0}};
            index[TIMEDIM] = time;
            index[SECTIONDIM] = indexcount;
            timeseries_out->SetPixel(index, (sum/count-average)/average);
            
            indexcount++;
        }
    }
    
    //write out image
    itk::ImageFileWriter< ImageTimeSeries >::Pointer writer = 
                itk::ImageFileWriter< ImageTimeSeries >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());

    //setup iterators
    writer->SetFileName(filename.c_str());
    writer->SetInput(timeseries_out);
    fprintf(stderr, "Writing Image %s\n", filename.c_str());
    writer->Update();
}

void writeSectionsTimeseries(std::list< SectionType >& active_voxels, 
            std::string prefix, double average, itk::MetaDataDictionary& dict)
{
    int label;
    int slice;
    double sum;
    size_t count;
    std::ostringstream oss;
    Image4DType::SpacingType space4;
    space4[0] = 1.0;
    space4[1] = 1.0;
    space4[2] = 1.0;
    
    std::list< ImageTimeSeries::Pointer > timeseries_out;
    
    //put all the time iterators back to 0
    resetVoxels(active_voxels);
    
    //calculate the number of slices each region has
    std::list< SectionType >::iterator list_it = active_voxels.begin();
    std::vector< unsigned int > slicecount;
    slice = list_it->point.GetIndex()[2];
    label = list_it->label;
    count = 1;
    while(list_it != active_voxels.end()) {
        if(label != list_it->label) {
#ifdef DEBUG
            fprintf(stderr, "Region: %i, Count: %zu\n", label, count);
#endif// DEBUG
            slicecount.push_back(count);
            count=1;
            label = list_it->label;
            slice = list_it->point.GetIndex()[2];
        }
        if(slice != list_it->point.GetIndex()[2]) {
            count++;
            slice = list_it->point.GetIndex()[2];
        } 
        list_it++;
    }
    //push the last count on the back, since the last change isn't between
    //sections but at the end of the list
    slicecount.push_back(count);

    unsigned int slicecount_i = 0; //used to iterate through slicecount vector
    unsigned int sliceselect = 0; //used to iterate through slices within a section

//    {
//    double tmp;
//    itk::ExposeMetaData<double>(dict, "TemporalResolution", tmp);
//    fprintf(stderr, "res: %f\n", tmp);
//    }
    
    list_it = active_voxels.begin();
    //sum up each section for each time period
    //while none of the active_voxel iterators have reached the end
    while(!active_voxels.begin()->point.IsAtEndOfLine()) {
        //each line corresponds to a trip through active_voxels at a particular
        //  time. As the iterator moves through it steps time for each voxel
        list_it = active_voxels.begin();
        slicecount_i = 0;
        sliceselect = 0;
        std::list< ImageTimeSeries::Pointer >::iterator image_it = 
                    timeseries_out.begin();
        //each iteration through this loop represents a single slice or section
        //in the active_voxels list faster by slice then slower by section:
        //section,slice:
        //1,1 1,2 1,3 2,1 2,4 2,10 etc.
        while(list_it != active_voxels.end()) {
            //convenience variables
            label = list_it->label;
            slice = list_it->point.GetIndex()[2];
            int time = list_it->point.GetIndex()[3];
#ifdef DEBUG
            fprintf(stderr, "Label: %i Slice: %i Time: %i\n", label, slice, time);
#endif //DEBUG

            //used to get average over region
            sum = 0;
            count = 0;

            //since active_voxels is sorted by label/slice, once the label or slice
            //  changes it is time to move to the next section/slice
            while(list_it->label == label && 
                        list_it->point.GetIndex()[2] == slice) {
                sum += list_it->point.Get();
                ++(list_it->point); //move forward in time
                count++;
                list_it++;
            }


            //if image for this section does not exist add it
            if(image_it == timeseries_out.end()) {
                timeseries_out.push_back( initTimeSeries(
                            active_voxels.begin()->point.GetRegion().
                            GetSize()[3], //timelen
                            label, slicecount[slicecount_i]) ); //section, slice
                timeseries_out.back()->GetMetaDataDictionary()["TemporalResolution"]
                            = dict["TemporalResolution"];
                {
                    double tmp;
                    itk::ExposeMetaData(timeseries_out.back()->GetMetaDataDictionary()
                                , "TemporalResolution", tmp);
                    space4[3] = tmp;
                    timeseries_out.back()->SetSpacing(space4);
                }
                image_it = timeseries_out.end();
                image_it--;
            }

//            fprintf(stderr, "setting time: %i\n", time);

            //set the voxel in the timeseries equal to the normalized average
            //for the section/slice at the current time
            ImageTimeSeries::IndexType index = {{ 0,0,0,0}};
            index[TIMEDIM] = time;
            index[SLICEDIM] = sliceselect;
#ifdef DEBUG
            fprintf(stderr, "pos        : %lu %lu %lu %lu\n", index[0], index[1],
                        index[2], index[3]);
            fprintf(stderr, "size       : %lu %lu %lu %lu\n\n", 
                        (*image_it)->GetRequestedRegion().GetSize()[0],
                        (*image_it)->GetRequestedRegion().GetSize()[1],
                        (*image_it)->GetRequestedRegion().GetSize()[2],
                        (*image_it)->GetRequestedRegion().GetSize()[3]);
#endif //DEBUG
            (*image_it)->SetPixel(index, (sum/count-average)/average);
            oss.str("");
            oss << "slice " << sliceselect;
            itk::EncapsulateMetaData((*image_it)->GetMetaDataDictionary(), 
                        oss.str(), slice);
//            itk::EncapsulateMetaData(timeseries_out[image_i]->GetMetaDataDictionary(), 
//                        "section", label);
            
            //move forward in timeseries_out when a new label has been reached
            //also, obviously this must move forward in the slicecount array
            if(label != list_it->label) {
                image_it++;
                slicecount_i++;
                sliceselect = 0;
            } else {
                sliceselect++;
            }
        }
    }
    
    //write out images
    while(!timeseries_out.empty()) {
        itk::ImageFileWriter< ImageTimeSeries >::Pointer writer = 
            itk::ImageFileWriter< ImageTimeSeries >::New();
        writer->SetImageIO(itk::modNiftiImageIO::New());
    
        //setup iterators
        if(!timeseries_out.back()->GetMetaDataDictionary().HasKey("section")) {
            fprintf(stderr, "Error could not find metadata dictionary key "
                        "for section\n");
            exit(-5);
        }
        itk::ExposeMetaData(timeseries_out.back()->GetMetaDataDictionary(), 
                    "section", label);
        oss.str("");
        oss << prefix << "_" << label << ".nii.gz";
        writer->SetFileName(oss.str());
        writer->SetInput(timeseries_out.back());
        fprintf(stderr, "Writing Image %s\n", oss.str().c_str());
        writer->Update();
        timeseries_out.pop_back();
    }
}

void writeVolume(Image4DType::Pointer input, Image3DType::Pointer labels,
            Image3DType::Pointer mask, std::string filename)
{
    itk::ImageLinearIteratorWithIndex<Image3DType> label_it
                ( labels, labels->GetRequestedRegion() );
    itk::ImageLinearIteratorWithIndex<Image3DType> mask_it;

    if(mask.IsNotNull()) {
        mask_it = itk::ImageLinearIteratorWithIndex<Image3DType>
                    (mask, mask->GetRequestedRegion());
    }

    Image4DType::PointType phys_fmri; //stores a 4D point
    Image4DType::IndexType index4D = {{0, 0, 0, 31}};
    Image3DType::PointType phys_3D;   //stores a 3D point
    Image3DType::IndexType mask_index; //the index matching the fmri physical index
    Image3DType::IndexType label_index; //the index matching the fmri physical index
    
    itk::ImageLinearIteratorWithIndex<Image4DType> fmri_it
                ( input, input->GetRequestedRegion() );
    itk::ImageLinearIteratorWithIndex<Image4DType> end_it 
                ( input, input->GetRequestedRegion() );
    fmri_it.SetDirection(0);
    fmri_it.SetIndex(index4D);

    end_it.SetDirection(0);
    index4D[3] = 10;
    end_it.SetIndex(index4D);

    Image3DType::Pointer out = Image3DType::New();
    
    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;

    for(int i = 0 ; i < 4 ; i++) 
        out_size[i] = input->GetRequestedRegion().GetSize()[i];
    
    for(int i = 0 ; i < 4 ; i++) 
        out_index[i] = input->GetRequestedRegion().GetIndex()[i];
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    out->SetRegions( out_region );
    out->Allocate();
    
    itk::ImageLinearIteratorWithIndex<Image3DType> out_it 
                ( out, out->GetRequestedRegion() );
    out_it.SetDirection(0);
    out_it.GoToBegin();

    Image3DType::SpacingType space;
    for( int i = 0 ; i < 4 ; i++ )
        space[i] = input->GetSpacing()[i];
    
    Image3DType::DirectionType direc = labels->GetDirection();
//    direc(0, 0) = input->GetDirection()(0, 0);
//    direc(0, 1) = input->GetDirection()(0, 1);
//    direc(0, 2) = input->GetDirection()(0, 2);
//    direc(1, 0) = input->GetDirection()(1, 0);
//    direc(1, 1) = input->GetDirection()(1, 1);
//    direc(1, 2) = input->GetDirection()(1, 2);
//    direc(2, 0) = input->GetDirection()(2, 0);
//    direc(2, 1) = input->GetDirection()(2, 1);
//    direc(2, 2) = input->GetDirection()(2, 2);

    Image3DType::PointType origin = labels->GetOrigin();
    origin[0] = input->GetOrigin()[0];
    origin[1] = input->GetOrigin()[1];
    origin[2] = input->GetOrigin()[2];

    out->SetSpacing(space);
    out->SetOrigin(origin);
    out->SetDirection(direc);

    fprintf(stderr, "Out Orientation\n");
    for(int ii = 0 ; ii < 3 ; ii++) {
        for(int jj = 0 ; jj < 3 ; jj++) {
            fprintf(stderr, "%f ", out->GetDirection()(jj,ii));
        }
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Out Origin\n");
    for(int ii = 0 ; ii < 3 ; ii++)  {
        fprintf(stderr, "%f ", out->GetOrigin()[ii]);
    }
    fprintf(stderr, "\n");

    if(mask.IsNotNull()) {
        fprintf(stderr, "mask Orientation\n");
        for(int ii = 0 ; ii < 3 ; ii++) {
            for(int jj = 0 ; jj < 3 ; jj++) {
                fprintf(stderr, "%f ", mask->GetDirection()(jj,ii));
            }
        }
        fprintf(stderr, "\n");

        fprintf(stderr, "Mask Origin\n");
        for(int ii = 0 ; ii < 3 ; ii++)  {
            fprintf(stderr, "%f ", mask->GetOrigin()[ii]);
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "Label Orientation\n");
    for(int ii = 0 ; ii < 3 ; ii++) {
        for(int jj = 0 ; jj < 3 ; jj++) {
            fprintf(stderr, "%f ", labels->GetDirection()(jj,ii));
        }
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Label Origin\n");
    for(int ii = 0 ; ii < 3 ; ii++)  {
        fprintf(stderr, "%f ", labels->GetOrigin()[ii]);
    }
    fprintf(stderr, "\n");
    
    while(!out_it.IsAtEnd()) {
        while(!out_it.IsAtEndOfLine()) {
            out->TransformIndexToPhysicalPoint( out_it.GetIndex(), phys_3D);
//            phys_3D[0] = phys_fmri[0];
//            phys_3D[1] = phys_fmri[1];
//            phys_3D[2] = phys_fmri[2];
            if(labels->TransformPhysicalPointToIndex(phys_3D, label_index) &&
                        (mask.IsNull() || 
                        mask->TransformPhysicalPointToIndex(phys_3D, mask_index))) {
                
                if(mask.IsNotNull())
                    mask_it.SetIndex(mask_index);

                label_it.SetIndex(label_index);
                if(label_it.Get() != 0 && (mask.IsNull() || mask_it.Get() != 0 )) {
                    out_it.Value() = fmri_it.Value();
                } else {
                    out_it.Value() = 0;
                }
            }
            out_it.Value() = fmri_it.Value();
            ++fmri_it;
            ++out_it;
        }
        out_it.NextLine();
    }

    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(filename);  
    writer->SetInput(out);
    writer->Update();
    fprintf(stderr, "Saved as %s", filename.c_str());
}

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    
    vul_arg<string> a_fmridir(0 ,"Directory with fmri timeseries");
    vul_arg<string> a_labels(0 ,"Labelmap image");
    vul_arg<double> a_skip("-skip" ,"Amount of time to skip at the beginning", 0.);
    vul_arg<string> a_mask("-m", "Greymatter mask", "");
    vul_arg<string> a_volume("-v", "Name to save volume at t=10 to", "");
    vul_arg<string> a_fullout("-c", "Cloned output of fmri timeseries", "");
    vul_arg<string> a_timeseries("-t", "Timeseries file, sections X time", "");
    vul_arg<string> a_timeprefix("-tp", "Timeseries file prefix, with each section"
                " in a separate file and, slices X time", "");
    vul_arg<string> a_regionprefix("-rp", "Region file prefix, with each full"
                " region from the input in a separet 4D file. x X y x Z X time",
                "");
    vul_arg_parse(argc, argv);
    
    fprintf(stderr, "Reading Dicom Directory: %s...\n", a_fmridir().c_str());
    Image4DType::Pointer fmri_img = read_dicom(a_fmridir(), a_skip());
    fprintf(stderr, "Done reading\n");
    
    if(!a_fullout().empty()) {
        //create a 4D image writer to save the image of appropriate size.
        itk::ImageFileWriter< Image4DType >::Pointer writer4d = 
            itk::ImageFileWriter< Image4DType >::New();
        writer4d->SetImageIO(itk::modNiftiImageIO::New());
        writer4d->SetFileName(a_fullout());  
        writer4d->SetInput(fmri_img);
        writer4d->Update();
    }

    //label index
    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
                itk::ImageFileReader<Image3DType>::New();
    labelmap_read->SetFileName( a_labels() );
    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
    labelmap_img->Update();

    //mask file

    Image3DType::Pointer mask_img;
    if(!a_mask().empty()) {
        itk::ImageFileReader<Image3DType>::Pointer mask_read = 
                    itk::ImageFileReader<Image3DType>::New();
        mask_read->SetFileName( a_mask() );
        mask_img = mask_read->GetOutput();
        mask_img->Update();
    }

    std::list< SectionType > active_voxels;

    fprintf(stderr, "Grabbing Segments...\n");
    int num_sections = segment(fmri_img, labelmap_img, mask_img, active_voxels);
    fprintf(stderr, "Done with %i active sections.\n", num_sections);
                
//    std::list< SectionType >::iterator list_it = active_voxels.begin();
//    while(list_it != active_voxels.end()) {
//       fprintf(stderr, "%li %li %li\n", list_it->point.GetIndex()[0],
//               list_it->point.GetIndex()[1], list_it->point.GetIndex()[2]);
//       ++list_it;
//    }

    fprintf(stderr, "Averaging over time and space...\n");
    double average = get_average(active_voxels);
    fprintf(stderr, "Done with average: %f\n", average);

    if(!a_timeprefix().empty()) {
        fprintf(stderr, "Generating Time images\n");
        writeSectionsTimeseries(active_voxels, a_timeprefix(), average, 
                    fmri_img->GetMetaDataDictionary());
        fprintf(stderr, "Done.\n");
    }

    if(!a_timeseries().empty()) {
        fprintf(stderr, "Generating Time Image\n");
        writeTimeseries(active_voxels, a_timeseries(), average, 
                    fmri_img->GetMetaDataDictionary());
        fprintf(stderr, "Done.\n");
    }

    if(!a_volume().empty()) {
        fprintf(stderr, "Saving volume in middle for region checking purposes\n");
        writeVolume(fmri_img, labelmap_img, mask_img, a_volume());
        fprintf(stderr, "Done.\n");
    }
    
    if(!a_regionprefix().empty()) {
        fprintf(stderr, "Generating 4D FMRI image for each section.\n");
        writeSections4D(active_voxels, a_regionprefix(), average, 
                    fmri_img->GetMetaDataDictionary(), fmri_img);
        fprintf(stderr, "Done.\n");
    }

    return 0;
}

