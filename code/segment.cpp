//image readers
#include "itkImageFileReader.h"

//standard libraries
#include <ctime>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkMetaDataDictionary.h>
#include <itkNaryElevateImageFilter.h>
#include <itkMaskImageFilter.h>

#include "segment.h"
//#include <itkLabelStatisticsImageFilterMod.h>
#include <itkLabelStatisticsImageFilter.h>

#include <itkSubtractConstantFromImageFilter.h>
#include <itkDivideByConstantImageFilter.h>
#include <itkNaryAddImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkDivideByConstantImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

typedef itk::AddImageFilter< Image3DType > AddF;
typedef itk::MultiplyByConstantImageFilter< Image3DType, double, Image3DType > ScaleF;
typedef itk::BinaryThresholdImageFilter<Label3DType, Label3DType> ThreshF;
//typedef itk::LabelStatisticsImageFilterMod<Image3DType, Label3DType> StatF3D;
typedef itk::LabelStatisticsImageFilter<Image3DType, Label3DType> StatF3D;

typedef itk::MaskImageFilter<Image4DType, Label4DType, Image4DType> MaskF;
//typedef itk::LabelStatisticsImageFilterMod<Image4DType, Label4DType> StatF4D;
typedef itk::LabelStatisticsImageFilter<Image4DType, Label4DType> StatF4D;
typedef itk::ExtractImageFilter<Image4DType, Image3DType> ExtF;
typedef itk::NaryAddImageFilter< Image3DType, Image3DType > AddNF;
typedef itk::SubtractImageFilter< Image4DType > SubF;
typedef itk::SubtractConstantFromImageFilter< Image4DType, double, 
            Image4DType > SubCF;
typedef itk::DivideImageFilter< Image4DType, Image4DType, Image4DType > DivF;
typedef itk::DivideByConstantImageFilter< Image4DType, double, Image4DType > DivCF;

const int TIMEDIM = 3;
const int SECTIONDIM = 0;

//sort first by increasing label, then by increasing time
//then by increasing slice, then by dim[1] then dim[0]
bool compare_lt(SectionType first, SectionType second)
{
    if(first.label < second.label) 
        return true;
    if(first.label > second.label) 
        return false;
    for(int i = 3 ; i >= 0 ; i--) {
        if(first.point.GetIndex()[i] < second.point.GetIndex()[i]) return true;
        if(first.point.GetIndex()[i] > second.point.GetIndex()[i]) return false;
    }
    return false;
}

Image4DType::Pointer initTimeSeries(size_t tlen, int sections)
{
    //create a 4D output image of appropriate size.
    Image4DType::Pointer outputImage = Image4DType::New();

    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index = {{0,0,0,0}};
    Image4DType::SizeType out_size = {{1, 1, 1, 1}};
    
    out_size[SECTIONDIM] = sections;
    out_size[TIMEDIM] = tlen;
    fprintf(stderr, " numsection : %lu\n", out_size[SECTIONDIM]);
    fprintf(stderr, " tlen       : %lu\n", out_size[TIMEDIM]);
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "Dim3", 
                std::string("time"));
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "Dim0", 
                std::string("section"));
    return outputImage;
}

template <typename T>
typename itk::OrientedImage<T,4>::Pointer stretch(
            typename itk::OrientedImage<T,3>::Pointer in, int length)
{
    typedef typename itk::OrientedImage<T,3> I3D;
    typedef typename itk::OrientedImage<T,4> I4D;
    typedef typename itk::NaryElevateImageFilter< I3D, I4D > NaryF;
    
    typename NaryF::Pointer elevateFilter = NaryF::New();
//    typename itk::NaryElevateImageFilter< typename itk::OrientedImage<T,3>, 
//                typename itk::OrientedImage<T,4> >::Pointer elevateFilter =
//                itk::NaryElevateImageFilter < typename itk::OrientedImage<T,3>, 
//                typename itk::OrientedImage<T,4> >::New(); 
    for(int i=0; i < length ; i++)
        elevateFilter->PushBackInput(in);
    elevateFilter->Update();
    elevateFilter->GetOutput()->CopyInformation(in);
    elevateFilter->GetOutput()->SetMetaDataDictionary(in->GetMetaDataDictionary());
    return elevateFilter->GetOutput();
}

Image4DType::Pointer normalizeByVoxel(const Image4DType::Pointer fmri_img)
{
    /* Extract Filter, will remove the 4th dimension (time) */
    ExtF::Pointer extract;
    Image4DType::RegionType region = fmri_img->GetRequestedRegion();
    region.SetSize(3, 0);
    
    /* Used to zero out the addfilter */
    Image3DType::Pointer zero = Image3DType::New();
    Image3DType::RegionType region3d;
    for(int i=0 ; i < 3 ; i++)
        region3d.SetSize(i, region.GetSize(i));
    for(int i=0 ; i < 3 ; i++)
        region3d.SetIndex(i, region.GetIndex(i));
    zero->SetRequestedRegion(region3d);
    zero->Allocate();
    zero->FillBuffer(0);
    
    /* Initialize the Addition */
    AddF::Pointer add = AddF::New();
    add->GraftOutput(zero);
    add->SetInput2(add->GetOutput());
    
    /* Calculate Sum of Images */
    for(size_t ii = 0 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        region.SetIndex(3, ii);
        extract = ExtF::New();
        extract->SetInput(fmri_img);
        extract->SetExtractionRegion(region);
        add->SetInput1(extract->GetOutput());
        add->Update();
    }

    /* Calculate Average of Images */
    ScaleF::Pointer scale = ScaleF::New();
    scale->SetInput(add->GetOutput());
    scale->SetConstant(1./fmri_img->GetRequestedRegion().GetSize()[3]);
    scale->Update();

    /* Rescale (fmri - avg)/avg*/
    Image4DType::Pointer avg4d = stretch<DataType>(scale->GetOutput(), 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    SubF::Pointer sub = SubF::New();
    DivF::Pointer div = DivF::New();
    sub->SetInput1(fmri_img);
    sub->SetInput2(avg4d);
    div->SetInput1(sub->GetOutput());
    div->SetInput2(avg4d);
    div->Update();
    
    div->GetOutput()->CopyInformation(extract->GetOutput());
    div->GetOutput()->SetMetaDataDictionary(
                extract->GetOutput()->GetMetaDataDictionary() );
    return div->GetOutput();
}

Image4DType::Pointer normalizeByGlobal(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer label_img)
{
    ThreshF::Pointer thresh = ThreshF::New();
    StatF4D::Pointer stats = StatF4D::New();

    thresh->SetLowerThreshold(0);
    thresh->SetUpperThreshold(0);
    thresh->SetInsideValue(0);
    thresh->SetOutsideValue(1);
    thresh->SetInput(label_img);
    thresh->Update();

    Label4DType::Pointer mask = stretch<LabelType>(thresh->GetOutput(), 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    stats->SetLabelInput(mask);
    stats->SetInput(fmri_img);
    stats->Update();
    
    double mean = stats->GetMean(1);
    
    SubCF::Pointer sub = SubCF::New();
    DivCF::Pointer div = DivCF::New();
    sub->SetConstant(mean);
    sub->SetInput(fmri_img);

    div->SetConstant(mean);
    div->SetInput(sub->GetOutput());
    div->Update();
    
    div->GetOutput()->CopyInformation(fmri_img);
    div->GetOutput()->SetMetaDataDictionary(fmri_img->GetMetaDataDictionary());
    return div->GetOutput();
}

Image4DType::Pointer normalizeByRegion(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap)
{
    std::ostringstream oss;

    /* Convert labelmap to 4D and then use it to get stats */
    Label4DType::Pointer labelmap4d = stretch<LabelType>(labelmap, 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    StatF4D::Pointer totalstats = StatF4D::New();
    totalstats->SetLabelInput(labelmap4d);
    totalstats->SetInput(fmri_img);
    totalstats->Update();

    Image4DType::Pointer output = initTimeSeries(
                fmri_img->GetRequestedRegion().GetSize()[3],
                totalstats->GetNumberOfLabels());
    
    /* Extract Filter, will remove the 4th dimension (time) */
    ExtF::Pointer extract;
    Image4DType::RegionType region = fmri_img->GetRequestedRegion();
    region.SetSize(3, 0);
    
    /* Calculate the average for each label at each timepoint */
    StatF3D::Pointer stats = StatF3D::New();
    stats->SetLabelInput(labelmap);

    /* Setup mapping of labels to indexes, and go ahead and add the first
     * time step to the output image
     */
    region.SetIndex(3,0);
    extract = ExtF::New();
    extract->SetInput(fmri_img);
    extract->SetExtractionRegion(region);
    
    stats->SetInput(extract->GetOutput());
    stats->Update();

    double result = 0;
    int label = 0;
    int index = 0;
    Image4DType::IndexType fullindex = {{index, 0, 0, 0}};
    
//    StatF3D::MapIterator it = stats->begin();
//    while(it != stats->end()) {
//        label = it->first;
//        if(label != 0) {
//            /* Map the section to the index */
//            oss.str("");
//            oss << "section:" << std::setfill('0') << std::setw(5) << label;
//            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
//                        index);
//            
//            /* Map the index to the section */
//            oss.str("");
//            oss << "index:" << std::setfill('0') << std::setw(5) << index;
//            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
//                        label);
//            
//            /* Fill in the data in the output */
//            result = (stats->GetMean(label)-totalstats->GetMean(label))/
//                        totalstats->GetMean(label);
//            fullindex[SECTIONDIM] = index;
//            output->SetPixel( fullindex, result);
//            index++;
//        }
//    }
    
    /* For the rest of the timesteps the index/sections are already mapped */
    for(size_t ii = 1 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        region.SetIndex(3, ii);
        extract = ExtF::New();
        extract->SetInput(fmri_img);
        extract->SetExtractionRegion(region);
        
        stats->SetInput(extract->GetOutput());
        stats->Update();
            
        fullindex[TIMEDIM] = ii;

        /* Write out the normalized value of each label */
//        StatF3D::MapIterator it = stats->begin();
//        while(it != stats->end()) {
//            label = it->first;
//            if(label != 0) {
//                oss.str("");
//                oss << "section:" << std::setfill('0') << std::setw(5) << label;
//                itk::ExposeMetaData(output->GetMetaDataDictionary(), oss.str(),
//                            index);
//            
//                /* Fill in the data in the output */
//                result = (stats->GetMean(label)-totalstats->GetMean(label))/
//                        totalstats->GetMean(label);
//                fullindex[SECTIONDIM] = index;
//                output->SetPixel(fullindex, result);
//            }
//        }
        
    }

    return output;
}

Image4DType::Pointer applymask(const Image4DType::Pointer fmri_img, 
            const Label3DType::Pointer mask_img)
{
    Label4DType::Pointer mask_img4d = stretch<LabelType>(mask_img, 
                fmri_img->GetRequestedRegion().GetSize()[3]);

    itk::MaskImageFilter<Image4DType, Label4DType, Image4DType>::Pointer mask =
                itk::MaskImageFilter<Image4DType, Label4DType, Image4DType>::New();
    mask->SetInput1(fmri_img);
    mask->SetInput2(mask_img4d);
    mask->Update();

    return mask->GetOutput();
}

//Reads a dicom directory then returns a pointer to the image
//does some of this memory need to be freed??
Image4DType::Pointer read_dicom(std::string directory, double skip)
{
    // create elevation filter
    itk::NaryElevateImageFilter<Image3DType, Image4DType>::Pointer elevateFilter
                = itk::NaryElevateImageFilter<Image3DType, Image4DType>::New();

    
    // create name generator and attach to reader
    itk::GDCMSeriesFileNames::Pointer nameGenerator = itk::GDCMSeriesFileNames::New();
    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0020|0100"); // acquisition number
    nameGenerator->SetDirectory(directory);

    // get series IDs
    const std::vector<std::string>& seriesUID = nameGenerator->GetSeriesUIDs();

    // create reader array
    itk::ImageSeriesReader<Image3DType>::Pointer *reader = 
                new itk::ImageSeriesReader<Image3DType>::Pointer[seriesUID.size()];

    // declare series iterators
    std::vector<std::string>::const_iterator seriesItr=seriesUID.begin();
    std::vector<std::string>::const_iterator seriesEnd=seriesUID.end();
    
    Image4DType::SpacingType space4;
    Image4DType::DirectionType direc; //c = labels->GetDirection();
    Image4DType::PointType origin; //c = labels->GetDirection();

    double temporalres = 2;
    //reorder the input based on temporal number.
    while (seriesItr!=seriesEnd)
    {
        itk::ImageSeriesReader<Image3DType>::Pointer tmp_reader = 
                    itk::ImageSeriesReader<Image3DType>::New();

        itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
        tmp_reader->SetImageIO(dicomIO);

        std::vector<std::string> fileNames;

//        printf("Accessing %s\n", seriesItr->c_str());
        fileNames = nameGenerator->GetFileNames(seriesItr->c_str());
        tmp_reader->SetFileNames(fileNames);

        tmp_reader->ReleaseDataFlagOn();

        tmp_reader->Update();
        tmp_reader->GetOutput()->SetMetaDataDictionary(
                    dicomIO->GetMetaDataDictionary());
    
        itk::MetaDataDictionary& dict = 
                    tmp_reader->GetOutput()->GetMetaDataDictionary();
        
        std::string value;
        dicomIO->GetValueFromTag("0020|0100", value);
        printf("Temporal Number: %s\n", value.c_str());
        
        reader[atoi(value.c_str())-1] = tmp_reader;

        dicomIO->GetValueFromTag("0018|0080", value);
        temporalres = atof(value.c_str())/1000.;
        itk::EncapsulateMetaData(dict, "TemporalResolution", temporalres);
    
        ++seriesItr;
    }

    for(int ii = 0 ; ii < 3 ; ii++)
        space4[ii] = reader[0]->GetOutput()->GetSpacing()[ii];
    space4[3] = temporalres;

    for(int ii = 0 ; ii<4 ; ii++) {
        for( int jj=0; jj<4 ; jj++) {
            if(ii == 3 || jj == 3) 
                direc(ii, jj) = 0;
            else
                direc(ii, jj) = reader[0]->GetOutput()->GetDirection()(ii, jj);
        }
    }
    direc(3, 3) = 1;
        
    // connect each series to elevation filter
    // skip the first two times
    for(unsigned int ii=0; ii<seriesUID.size(); ii++) {
        if(ii*temporalres >= skip) {
            elevateFilter->PushBackInput(reader[ii]->GetOutput());
        } else {
            fprintf(stderr, "Skipping: %i, %f\n", ii, ii*temporalres);
        }
    }
        
    //now elevateFilter can be used just like any other reader
    elevateFilter->Update();
    
    //create image readers
    Image4DType::Pointer fmri_img = elevateFilter->GetOutput();
//    fmri_img->CopyInformation(reader[0]->GetOutput());
    fmri_img->SetDirection(direc);
    fmri_img->SetSpacing(space4);
    fmri_img->Update();
    
    delete[] reader;
    return fmri_img;
};

//example main:
//The labelmap should already have been masked through a maxprob image for
//graymatter
//int main( int argc, char **argv ) 
//{
//    // check arguments
//    if(argc != 3) {
//        printf("Usage: %s <4D fmri dir> <labels>", argv[0]);
//        return EXIT_FAILURE;
//    }
//    
//    Image4DType::Pointer fmri_img = read_dicom(argv[1]);
//
//    //label index
//    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
//                itk::ImageFileReader<Image3DType>::New();
//    labelmap_read->SetFileName( argv[2] );
//    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
//    labelmap_img->Update();
//
//    std::list< SectionType* > active_voxels;
//
//    sort_voxels(fmri_img, labelmap_img, active_voxels);
//    
//    return 0;
//}

