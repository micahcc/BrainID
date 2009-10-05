#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"

#include "modNiftiImageIO.h"

#include <sstream>
#include <iostream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

typedef itk::OrientedImage<double, 4> ImageTimeSeries;

using namespace std;


void writeVolume(Image4DType::Pointer input, std::string filename, int index)
{
    Image3DType::Pointer out = extract(input, index);
    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(filename);  
    writer->SetInput(out);
    writer->Update();
    fprintf(stderr, "Saved as %s\n", filename.c_str());
}

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_fmridir(0 ,"Directory with fmri timeseries");
    vul_arg<string> a_fullout(0 ,"Cloned output of fmri timeseries");
    
    /* Output related */
    vul_arg<string> a_volume("-v", "Name to save volume at t=10 to", "");

    vul_arg<double> a_skip("-s", "Amount of time to skip at front", 0);
    vul_arg_parse(argc, argv);
    
    ostringstream oss;

    fprintf(stderr, "Reading Dicom Directory: %s...\n", a_fmridir().c_str());
    Image4DType::Pointer fmri_img = read_dicom(a_fmridir(), a_skip());
    fprintf(stderr, "Done reading\n");
    
    {
        //create a 4D image writer to save the image of appropriate size.
        itk::ImageFileWriter< Image4DType >::Pointer writer4d = 
            itk::ImageFileWriter< Image4DType >::New();
        writer4d->SetImageIO(itk::modNiftiImageIO::New());
        writer4d->SetFileName(a_fullout());  
        writer4d->SetInput(fmri_img);
        writer4d->Update();
    }
    
    if(!a_volume().empty()) {
        fprintf(stderr, "Saving volume in middle for region checking purposes\n");
        oss.str("");
        oss << a_volume() << "-10.nii.gz";
        writeVolume(fmri_img, oss.str(), 10);
        
        oss.str("");
        oss << a_volume() << "-20.nii.gz";
        writeVolume(fmri_img, oss.str(), 20);
        
        oss.str("");
        oss << a_volume() << "-30.nii.gz";
        writeVolume(fmri_img, oss.str(), 30);
        fprintf(stderr, "Done.\n");
    }

    unsigned int skip = 0;
    itk::ExposeMetaData(fmri_img->GetMetaDataDictionary(),
                "offset", skip);
    fprintf(stderr, "Note that the timeseries had %f seconds removed from the "
                "beginning\n", skip*fmri_img->GetSpacing()[3]);

    return 0;
}

