#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"
#include "tools.h"

#include <itkMaskImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include "modNiftiImageIO.h"

#include <sstream>
#include <iostream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_input(0 ,"Input Image");
    vul_arg<string> a_label(0 ,"Label Image, mean will be over each label");
    vul_arg<string> a_output(0 ,"Output Image");
    vul_arg<unsigned int> a_spline("-spline" ,"knots to use for detrending", 5);
    
    /* Processing */
    vul_arg<string> a_filtered("-f", "Filtered/Normalized bold 4D Image", "");
    vul_arg<string> a_mask("-m" ,"Mask Image to apply to labelmap", "");

    vul_arg_parse(argc, argv);

    if(a_spline() < 3) {
        fprintf(stderr, "Error need at least 3 knots\n");
        return -4;
    }
    
    Image4DType::Pointer fmri_img;
    {
    //FMRI Image
    itk::ImageFileReader<Image4DType>::Pointer fmri_read = 
                itk::ImageFileReader<Image4DType>::New();
    fmri_read->SetFileName( a_input() );
    fmri_img = fmri_read->GetOutput();
    fmri_img->Update();
    }
    
    //label index
    Label3DType::Pointer labelmap_img;
    {
    itk::ImageFileReader<Label3DType>::Pointer labelmap_read = 
                itk::ImageFileReader<Label3DType>::New();
    labelmap_read->SetFileName( a_label() );
    labelmap_img = labelmap_read->GetOutput();
    labelmap_img->Update();
    }

    //mask file
    if(!a_mask().empty()) {
        Label3DType::Pointer mask_img;
        itk::ImageFileReader<Label3DType>::Pointer mask_read = 
                    itk::ImageFileReader<Label3DType>::New();
        mask_read->SetFileName( a_mask() );
        
        fprintf(stderr, "Applying mask to label...\n");
        itk::MaskImageFilter< Label3DType, Label3DType, Label3DType >::Pointer maskf =
                    itk::MaskImageFilter< Label3DType, Label3DType, Label3DType >::New();
        maskf->SetInput1(labelmap_img);
        maskf->SetInput2(mask_read->GetOutput());
        maskf->Update();
        labelmap_img = maskf->GetOutput();
    }


    fprintf(stderr, "Normalizing by voxel...\n");
    fmri_img = normalizeByVoxel(fmri_img, labelmap_img, a_spline());
    fprintf(stderr, "Done\n");
       
    cout << "filtered " <<  a_filtered() << " " << a_filtered().empty() << endl;
    if(!a_filtered().empty()) {
        fprintf(stderr, "Writing Filtered Image\n");

        itk::ImageFileWriter< Image4DType >::Pointer writer = 
                    itk::ImageFileWriter< Image4DType >::New();
        writer->SetInput(fmri_img);
        writer->SetImageIO(itk::modNiftiImageIO::New());
        writer->SetFileName(a_filtered());
        writer->Update();
        fprintf(stderr, "Done.\n");
    }
        
    Image4DType::Pointer timeseries;
    fprintf(stderr, "Generating Timeseries...\n");
    std::list<LabelType> labels;
    timeseries = summ(fmri_img, labelmap_img, labels);
    fprintf(stderr, "Done\n");

    cout << "time " <<  a_output() << " " << a_output().empty() << endl;
    if(!a_output().empty()) {
        fprintf(stderr, "Writing Time Image\n");

        itk::ImageFileWriter< Image4DType >::Pointer writer = 
                    itk::ImageFileWriter< Image4DType >::New();
        unsigned int offset = 0;
        itk::ExposeMetaData(timeseries->GetMetaDataDictionary(), "offset", offset);
        fprintf(stderr, "Offset: %u", offset);
        writer->SetInput(timeseries);
        writer->SetImageIO(itk::modNiftiImageIO::New());
        writer->SetFileName(a_output());
        writer->Update();
        fprintf(stderr, "Done.\n");
    }
        
    return 0;
}


