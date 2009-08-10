//image readers
#include "itkImageFileReader.h"

//standard libraries
#include <ctime>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include <itkImageFileWriter.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkMetaDataDictionary.h>
#include <itkNaryElevateImageFilter.h>
#include <itkMaskImageFilter.h>

#include "segment.h"
#include <itkLabelStatisticsImageFilterMod.h>
#include <itkCastImageFilter.h>
#include <itkImageLinearIteratorWithIndex.h>

#include <itkSubtractConstantFromImageFilter.h>
#include <itkDivideByConstantImageFilter.h>
#include <itkNaryAddImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkDivideByConstantImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

#include <gsl/gsl_spline.h>

typedef itk::AddImageFilter< Image3DType > AddF;
typedef itk::MultiplyByConstantImageFilter< Image3DType, double, Image3DType > ScaleF;
typedef itk::BinaryThresholdImageFilter<Label3DType, Label3DType> ThreshF;
typedef itk::LabelStatisticsImageFilterMod<Image3DType, Label3DType> StatF3D;
typedef itk::MaskImageFilter<Image3DType, Label3DType, Image3DType> MaskF;
//typedef itk::LabelStatisticsImageFilter<Image3DType, Label3DType> StatF3D;

typedef itk::LabelStatisticsImageFilterMod<Image4DType, Label4DType> StatF4D;
//typedef itk::LabelStatisticsImageFilter<Image4DType, Label4DType> StatF4D;
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
//bool compare_lt(SectionType first, SectionType second)
//{
//    if(first.label < second.label) 
//        return true;
//    if(first.label > second.label) 
//        return false;
//    for(int i = 3 ; i >= 0 ; i--) {
//        if(first.point.GetIndex()[i] < second.point.GetIndex()[i]) return true;
//        if(first.point.GetIndex()[i] > second.point.GetIndex()[i]) return false;
//    }
//    return false;
//}

/* Remove any elements that arent' in the reference list 
 * ref will be sorted, but otherwise will be unchanged*/
void removeMissing(std::list<LabelType>& ref, std::list<LabelType>& mod)
{
    ref.sort();
    mod.sort();
    
    std::list<LabelType>::iterator itref = ref.begin();
    std::list<LabelType>::iterator itmod = mod.begin();
    while(itmod != mod.end()) {
        if(*itmod == *itref) {
            itmod++;
            itref++;
        } else if(*itmod < *itref) {
            itmod = mod.erase(itmod);
        } else if(itref != ref.end()){
            itref++;
        } else {
            itmod = mod.erase(itmod);
        }
    }
}

template<class T, unsigned int SIZE>
void outputinfo(Image4DType::Pointer in) {
    fprintf(stderr, "Dimensions:\n");
    for(size_t ii = 0 ; ii < SIZE ; ii++)
        fprintf(stderr, "%zu ", in->GetRequestedRegion().GetSize()[ii]);
    
    fprintf(stderr, "\nIndex:\n");
    for(size_t ii = 0 ; ii < SIZE ; ii++)
        fprintf(stderr, "%zu ", in->GetRequestedRegion().GetIndex()[ii]);

    fprintf(stderr, "\nSpacing:\n");
    for(size_t ii = 0 ; ii < SIZE ; ii++)
        fprintf(stderr, "%f ", in->GetSpacing()[ii]);
    
    fprintf(stderr, "\nOrigin:\n");
    for(size_t ii = 0 ; ii < SIZE ; ii++)
        fprintf(stderr, "%f ", in->GetOrigin()[ii]);

    fprintf(stderr, "\nDirection:\n");
    for(size_t ii = 0 ; ii<SIZE ; ii++) {
        for( size_t jj=0; jj<SIZE ; jj++) {
            fprintf(stderr, "%f ", in->GetDirection()(ii,jj));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

Image3DType::Pointer extract(Image4DType::Pointer input, size_t index)
{
    Image4DType::IndexType index4D = {{0, 0, 0, 0}};
    
    itk::ImageLinearIteratorWithIndex<Image4DType> fmri_it
                ( input, input->GetRequestedRegion() );
    index4D[3] = index;
    fmri_it.SetDirection(0);
    fmri_it.SetIndex(index4D);

    Image3DType::Pointer out = Image3DType::New();
    
    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;
    
    Image3DType::SpacingType space;
    Image3DType::DirectionType direc;
    Image3DType::PointType origin; 

    for(int i = 0 ; i < 3 ; i++) 
        out_size[i] = input->GetRequestedRegion().GetSize()[i];
    
    for(int i = 0 ; i < 3 ; i++) 
        out_index[i] = input->GetRequestedRegion().GetIndex()[i];
    
    for(int i = 0 ; i < 3 ; i++) 
        space[i] = input->GetSpacing()[i];
    
    for(int i = 0 ; i < 3 ; i++) 
        origin[i] = input->GetOrigin()[i];
    
    for(int ii = 0 ; ii<3 ; ii++) {
        for(int jj = 0 ; jj<3 ; jj++) {
            direc(ii, jj) = input->GetDirection()(ii, jj);
        }
    }
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    out->SetRegions( out_region );

    out->SetDirection(direc);
    out->SetSpacing(space);
    out->SetOrigin(origin);

    out->Allocate();
    
    itk::ImageLinearIteratorWithIndex<Image3DType> out_it 
                ( out, out->GetRequestedRegion() );
    out_it.SetDirection(0);
    out_it.GoToBegin();

    while(!out_it.IsAtEnd()) {
        while(!out_it.IsAtEndOfLine()) {
            out_it.Value() = fmri_it.Value();
            ++fmri_it;
            ++out_it;
        }
        out_it.NextLine();
        fmri_it.NextLine();
    }
    
    return out;
}

template<class T, unsigned int SIZE1, unsigned int SIZE2>
void copyInformation(typename itk::OrientedImage<T, SIZE1>::Pointer in1, 
            typename itk::OrientedImage<T, SIZE2>::Pointer in2)
{
    typename itk::OrientedImage<T, SIZE2>::SpacingType space;
    typename itk::OrientedImage<T, SIZE2>::DirectionType direc; 
    typename itk::OrientedImage<T, SIZE2>::PointType origin; 
    for(size_t ii = 0 ; ii < SIZE2; ii++) {
        if(ii >= SIZE1)
            space[ii] = 1;
        else
            space[ii] = in1->GetSpacing()[ii];
    }

    for(size_t ii = 0 ; ii<SIZE2 ; ii++) {
        for( size_t jj=0; jj<SIZE2 ; jj++) {
            if(ii >= SIZE1 || jj >= SIZE1) {
                if(ii == jj) 
                    direc(ii, jj) = 1;
                else 
                    direc(ii, jj) = 0;
            } else {
                direc(ii, jj) = in1->GetDirection()(ii, jj);
            }
        }
    }
    
    for(size_t ii = 0 ; ii < SIZE2; ii++) {
        if(ii >= SIZE1)
            origin[ii] = 0;
        else
            origin[ii] = in1->GetOrigin()[ii];
    }
}

Image4DType::Pointer initTimeSeries(Image4DType::Pointer fmri_img, int sections)
{       
    //create a 4D output image of appropriate size.
    Image4DType::Pointer outputImage = Image4DType::New();

    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index = {{0,0,0,0}};
    Image4DType::SizeType out_size = {{1, 1, 1, 1}};
    
    out_size[SECTIONDIM] = sections;
    out_size[TIMEDIM] = fmri_img->GetRequestedRegion().GetSize()[3];
    fprintf(stderr, " numsection : %lu\n", out_size[SECTIONDIM]);
    fprintf(stderr, " tlen       : %lu\n", out_size[TIMEDIM]);
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    outputImage->FillBuffer(0);
    outputImage->SetMetaDataDictionary(fmri_img->GetMetaDataDictionary());
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "Dim3", 
                std::string("time"));
    itk::EncapsulateMetaData(outputImage->GetMetaDataDictionary(), "Dim0", 
                std::string("section"));
    outputImage->CopyInformation(fmri_img);

    return outputImage;
}

std::list<LabelType> getlabels(Label3DType::Pointer labelmap)
{
    /* Just used to acquire the list of labels */
    itk::LabelStatisticsImageFilterMod<Label3DType, Label3DType>::Pointer
                stats = 
                itk::LabelStatisticsImageFilterMod<Label3DType, Label3DType>::New();
    
    stats->SetLabelInput(labelmap);
    stats->SetInput(labelmap);
    stats->Update();
    return stats->GetLabels();
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
    itk::MetaDataDictionary dict = in->GetMetaDataDictionary();
    itk::MetaDataDictionary dict2;
    in->SetMetaDataDictionary(dict2);
    for(int i=0; i < length ; i++)
        elevateFilter->PushBackInput(in);
    elevateFilter->Update();
    copyInformation<T, 3, 4>(in, elevateFilter->GetOutput());
    elevateFilter->GetOutput()->SetMetaDataDictionary(dict);
    return elevateFilter->GetOutput();
}

int detrend(Image4DType::Pointer fmri_img, Image4DType::IndexType index, int regions)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, regions+2);

    for(int i = 0 ; i < 4 ; i++) {
        fprintf(stderr, "%zu ", index[i]);
    }
    fprintf(stderr, "\n");

    /* Go to index and start at time 0 at that voxel*/
    itk::ImageLinearIteratorWithIndex< Image4DType > fmri_it;
    fmri_it.SetDirection(3);
    fmri_it.SetIndex(index);

    double averages[regions+2];
    int counts[regions+2];
    int starts[regions];

    int length = fmri_img->GetRequestedRegion().GetSize()[3];
    double rsize = length / regions;
    for(int i = 0 ; i < regions ; i++) {
        starts[i] = rsize*i;
    }
    
    int region = 0;
    for(fmri_it.GoToBeginOfLine(); !fmri_it.IsAtEndOfLine(); ++fmri_it) {
        /* Figure out Region */
        if(fmri_it.GetIndex()[3] < length/(regions*2)) {
            averages[0] += fmri_it.Get();
            counts[0]++;
        } else if(fmri_it.GetIndex()[3] > (length - length/(regions*2))) {
            averages[regions+1] += fmri_it.Get();
            counts[regions+1]++;
        }

        /* Check to see if the index is for the last region(since there is no)
         * start for the region after that */
        if(fmri_it.GetIndex()[3] > starts[regions-1]) {
            averages[regions] += fmri_it.Get();
            counts[regions]++;
        /* Check to see if the index is for the next region and
         * if that is the case, then add the data to the next region
         * and iterate the region count*/
        } else if(fmri_it.GetIndex()[3] > starts[region+1]) {
            averages[region+2] += fmri_it.Get();
            counts[region+2]++;
            region++;
        /* Otherwise, just go with the current region */
        } else {
            averages[region+1] += fmri_it.Get();
            counts[region+1]++;
        }
    }

    double xpos[regions+2];
    xpos[0] = 0;
    xpos[regions+1] = length-1;;
    xpos[regions] = (length + starts[regions-1])/2;

    gsl_spline_init(spline, xpos, averages, regions+2);
    for(fmri_it.GoToBeginOfLine(); !fmri_it.IsAtEndOfLine(); ++fmri_it) {
        fmri_it.Set(gsl_spline_eval(spline, fmri_it.GetIndex()[3], acc));
    }

    for(int i = 1 ; i < regions ; i ++){
        xpos[i] = (starts[i+1] + starts[i])/2;
    }

    for(int i = 0 ; i  < regions+2 ; i++) {
        fprintf(stderr, "%i Average: %f Count: %i, pos: %f\n", i, averages[i], 
                    counts[i], xpos[i]);
    }
    for(int i = 0 ; i  < regions ; i++) {
        fprintf(stderr, "%i Starts: %i\n", i, starts[i]);
    }
    return 0;
}

Image4DType::Pointer normalizeByVoxel(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer mask, int regions)
{
    /* Fmri Iterators */
    itk::ImageLinearIteratorWithIndex< Image4DType > fmri_it;
    fmri_it.SetDirection(0);
    
    itk::ImageLinearIteratorWithIndex< Image4DType > fmri_stop;
    fmri_stop.SetDirection(3);
    fmri_stop.GoToBegin();
    ++fmri_stop;
    
    itk::ImageLinearIteratorWithIndex< Label3DType > mask_it;

    Image4DType::PointType point4;
    Label3DType::IndexType index3;
    Label3DType::PointType point3;
    
    for(fmri_it.GoToBegin(); fmri_it != fmri_stop ; fmri_it.NextLine()) {
        for( ; !fmri_it.IsAtEndOfLine(); ++fmri_it) {

            /* Change 4D Index in fmri Image to 3d in mask, which have spacing */
            fmri_img->TransformIndexToPhysicalPoint(fmri_it.GetIndex(), point4);
            for(int i = 0 ; i < 3 ; i++) point3[i] = point4[i];
            mask->TransformPhysicalPointToIndex(point3, index3);
            mask_it.SetIndex(index3);

            /* Re-write time series based on spline detrending */
            if(mask_it.Get() != 0) {
                detrend(fmri_img, fmri_it.GetIndex(), regions);
            }
        }
    }
    
    Image3DType::Pointer average = get_average(fmri_img);
    double globalmean = get_average(fmri_img, mask);
    /* Rescale (fmri - avg)/avg*/
    Image4DType::Pointer avg4d = stretch<DataType>(average, 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    avg4d->CopyInformation(fmri_img);
    
    DivCF::Pointer div = DivCF::New();
    div->SetInput(fmri_img);
    div->SetConstant(globalmean);
    div->Update();
    
    div->GetOutput()->CopyInformation(fmri_img);
    div->GetOutput()->SetMetaDataDictionary(
                fmri_img->GetMetaDataDictionary() );
    return div->GetOutput();
}

Image3DType::Pointer get_average(const Image4DType::Pointer fmri_img)
{
    /* Used to zero out the addfilter */
    Image3DType::Pointer zero = extract(fmri_img, 0);
    zero->FillBuffer(0);
    
    /* Initialize the Addition */
    AddF::Pointer add = AddF::New();
    add->GraftOutput(zero);
    add->SetInput2(add->GetOutput());
    
    /* Calculate Sum of Images */
    for(size_t ii = 0 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        add->SetInput1(extract(fmri_img, ii));
        add->Update();
    }

    /* Calculate Average of Images */
    ScaleF::Pointer scale = ScaleF::New();
    scale->SetInput(add->GetOutput());
    scale->SetConstant(1./fmri_img->GetRequestedRegion().GetSize()[3]);
    scale->Update();
    return scale->GetOutput();
}

double get_average(const Image4DType::Pointer fmri_img, 
        const Label3DType::Pointer labelmap)
{
    Image3DType::Pointer perVoxAvg = get_average(fmri_img);

    /* Change mask to 0/1 */
    ThreshF::Pointer thresh = ThreshF::New();
    thresh->SetLowerThreshold(0);
    thresh->SetUpperThreshold(0);
    thresh->SetInsideValue(0);
    thresh->SetOutsideValue(1);
    thresh->SetInput(labelmap);
    thresh->Update();

    /******************************************************* 
     * Calculate the average for everything inside the mask
     */
    StatF3D::Pointer stats = StatF3D::New();
    stats->SetLabelInput(thresh->GetOutput());
    stats->SetInput(perVoxAvg);
    stats->Update();
    
    return stats->GetMean(1);
}

Image4DType::Pointer normalizeByGlobal(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer label_img)
{
    double mean = get_average(fmri_img, label_img);
    
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

Image4DType::Pointer summGlobalNorm(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, std::list<LabelType>& labels)
{
    std::ostringstream oss;

    double mean = get_average(fmri_img, labelmap); 
    /******************************************************* 
     * Calculate the average for each label at each timepoint
     */
    StatF3D::Pointer stats = StatF3D::New();
    stats->SetLabelInput(labelmap);

    /* Setup mapping of labels to indexes, and go ahead and add the first
     * time step to the output image
     */
    stats->SetInput(extract(fmri_img, 0));
    stats->Update();
    
    std::list<LabelType> rlabels = stats->GetLabels();
    if(!labels.empty()) {
        removeMissing(labels, rlabels);
    }
    
    Image4DType::Pointer output = initTimeSeries(fmri_img, rlabels.size());

    double result = 0;
    int label = 0;
    int index = 0;
    Image4DType::IndexType fullindex = {{index, 0, 0, 0}};

    for(std::list<LabelType>::iterator it = rlabels.begin() ; 
                    it != rlabels.end(); it++) {
        label = *it;
        if(label != 0) {
            /* Map the section to the index */
            oss.str("");
            oss << "section:" << std::setfill('0') << std::setw(5) << label;
            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
                        index);
            fprintf(stderr, "%s -> %i\n", oss.str().c_str(), index);
            
            /* Map the index to the section */
            oss.str("");
            oss << "index:" << std::setfill('0') << std::setw(5) << index;
            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
                        label);
            fprintf(stderr, "%s -> %i\n", oss.str().c_str(), label);
            fprintf(stderr, "%lu Voxels\n", stats->GetCount(label));
            
            /* Fill in the data in the output */
            result = (mean == 0) ? 0 : (stats->GetMean(label)-mean)/mean;
            fullindex[SECTIONDIM] = index;
            output->SetPixel( fullindex, result);
            index++;
        }
    }
    
    /* For the rest of the timesteps the index/sections are already mapped */
    for(size_t ii = 1 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        stats->SetInput(extract(fmri_img, ii));
        stats->Update();
            
        fullindex[TIMEDIM] = ii;

        /* Write out the normalized value of each label */
        for(std::list<LabelType>::iterator it=rlabels.begin() ; 
                        it!=rlabels.end(); it++) {
            label = *it;
            if(label != 0) {
                oss.str("");
                oss << "section:" << std::setfill('0') << std::setw(5) << label;
                itk::ExposeMetaData(output->GetMetaDataDictionary(), oss.str(),
                            index);
            
                /* Fill in the data in the output */
                result = (mean == 0) ? 0 : (stats->GetMean(label)-mean)/mean;
                fullindex[SECTIONDIM] = index;
                output->SetPixel(fullindex, result);
            }
        }
        
    }

    return output;
}

Image4DType::Pointer summ(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, std::list<LabelType>& labels)
{
    std::ostringstream oss;

    StatF3D::Pointer stats = StatF3D::New();
    stats->SetLabelInput(labelmap);

    /* **************************************************************
     * Initialize Image,
     * Get a list of labels that need to be added
     * Setup mapping of labels to indexes, 
     * go ahead and add the first time step to the output image
     */
    stats->SetInput(extract(fmri_img, 0));
    stats->Update();
    
    /* Find the common labels between the input and the actual */
    std::list<LabelType> rlabels = stats->GetLabels();
    if(!labels.empty())
        removeMissing(labels, rlabels);

    //initialize output
    Image4DType::Pointer output = initTimeSeries(fmri_img, rlabels.size());

    int label = 0;
    int index = 0;
    Image4DType::IndexType fullindex = {{index, 0, 0, 0}};
    
    for(std::list<LabelType>::iterator it = rlabels.begin() ; 
                    it != rlabels.end(); it++) {
        label = *it;
        if(label != 0) {
            /* Map the section to the index */
            oss.str("");
            oss << "section:" << std::setfill('0') << std::setw(5) << label;
            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
                        index);
            fprintf(stderr, "%s -> %i\n", oss.str().c_str(), index);
            
            /* Map the index to the section */
            oss.str("");
            oss << "index:" << std::setfill('0') << std::setw(5) << index;
            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
                        label);
            fprintf(stderr, "%s -> %i\n", oss.str().c_str(), label);
            fprintf(stderr, "%lu Voxels\n", stats->GetCount(label));
            
            /* Fill in the data in the output */
            fullindex[SECTIONDIM] = index;
            output->SetPixel( fullindex, stats->GetMean(label));
            index++;
        }
    }
    
    /*******************************************************************
     * For the rest of the timesteps the index/sections are already mapped */
    for(size_t ii = 1 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        stats->SetInput(extract(fmri_img, ii));
        stats->Update();
            
        fullindex[TIMEDIM] = ii;

        /* Write out the normalized value of each label */
        for(std::list<LabelType>::iterator it = rlabels.begin() ; 
                        it != rlabels.end(); it++) {
            label = *it;
            if(label != 0) {
                oss.str("");
                oss << "section:" << std::setfill('0') << std::setw(5) << label;
                itk::ExposeMetaData(output->GetMetaDataDictionary(), oss.str(),
                            index);
            
                /* Fill in the data in the output */
                fullindex[SECTIONDIM] = index;
                output->SetPixel(fullindex, stats->GetMean(label));
            }
        }
        
    }

    return output;
}


template<class T, unsigned int SIZE1, class U, unsigned int SIZE2>
typename itk::OrientedImage<T, SIZE1>::Pointer applymask(
            typename itk::OrientedImage<T, SIZE1>::Pointer input, 
            typename itk::OrientedImage<U, SIZE2>::Pointer mask)
{
    typedef itk::CastImageFilter< itk::OrientedImage<T, SIZE1>, 
                itk::OrientedImage<T, SIZE1> > CastF;
    typename CastF::Pointer cast = CastF::New();
    cast->SetInput(input);
    cast->Update();
    typename itk::OrientedImage<T, SIZE1>::Pointer recast = cast->GetOutput();
                
    typename itk::OrientedImage<U, SIZE2>::IndexType maskindex;
    typename itk::OrientedImage<U, SIZE2>::PointType maskpoint;
    typename itk::ImageLinearIteratorWithIndex<itk::OrientedImage<U, SIZE2> > maskit
                (mask, mask->GetRequestedRegion());
//    typename itk::OrientedImage<T, SIZE1>::IndexType imgindex;
    typename itk::OrientedImage<T, SIZE1>::PointType imgpoint;
    typename itk::ImageLinearIteratorWithIndex<itk::OrientedImage<T, SIZE1> > imgit
                (recast, recast->GetRequestedRegion());
    
    imgit.GoToBegin();

    while(!imgit.IsAtEnd()) {
        while(!imgit.IsAtEndOfLine()) {
            recast->TransformIndexToPhysicalPoint(imgit.GetIndex(), imgpoint);
            for(size_t ii = 0 ; ii < SIZE2; ii++) {
                if(ii >= SIZE1)
                    maskpoint[ii] = 0;
                else
                    maskpoint[ii] = imgpoint[ii];
            }
            mask->TransformPhysicalPointToIndex(maskpoint, maskindex);
            maskit.SetIndex(maskindex);
            /* The double negative is because != will be false for NaN */
            if(!(maskit.Get() != 0)) {
                imgit.Set(0);
            }
            ++imgit;
        }
        imgit.NextLine();
    }

    return recast;
}


/* ???? */
Image4DType::Pointer splitByRegion(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, int label)
{
    std::ostringstream oss;
    
    /* For each region, threshold the labelmap to just the desired region */
    /* Change mask to 0/1 */
    ThreshF::Pointer thresh = ThreshF::New();
    thresh->SetLowerThreshold(label);
    thresh->SetUpperThreshold(label);
    thresh->SetInsideValue(1);
    thresh->SetOutsideValue(0);
    thresh->SetInput(labelmap);
    thresh->Update();

    return applymask<DataType, 4, LabelType, 3>(fmri_img, thresh->GetOutput());
}

/* ???? */
Image4DType::Pointer summRegionNorm(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, std::list<LabelType>& sections)
{
    std::ostringstream oss;

    Image3DType::Pointer perVoxAvg = get_average(fmri_img);
    double globalmean = get_average(fmri_img, labelmap); 

    /* Take the average of the average of the voxels in each region */
    StatF3D::Pointer totalstats = StatF3D::New();
    totalstats->SetLabelInput(labelmap);
    totalstats->SetInput(perVoxAvg);
    totalstats->Update();

    /******************************************************* 
     * Calculate the average for each label at each timepoint
     */
    StatF3D::Pointer stats = StatF3D::New();
    stats->SetLabelInput(labelmap);

    /* Setup mapping of labels to indexes, and go ahead and add the first
     * time step to the output image
     */
    stats->SetInput(extract(fmri_img, 0));
    stats->Update();

    double result = 0;
    int label = 0;
    int index = 0;
    Image4DType::IndexType fullindex = {{index, 0, 0, 0}};

    std::list<LabelType> rlabels = stats->GetLabels();
    if(!sections.empty()) {
        removeMissing(sections, rlabels);
    }
    
    Image4DType::Pointer output = initTimeSeries(fmri_img, rlabels.size());
    
    for(std::list<LabelType>::iterator it=rlabels.begin() ;
                    it!=rlabels.end(); it++) {
        label = *it;
        if(label != 0) {
            /* Map the section to the index */
            oss.str("");
            oss << "section:" << std::setfill('0') << std::setw(5) << label;
            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
                        index);
            fprintf(stderr, "%s -> %i\n", oss.str().c_str(), index);
            
            /* Map the index to the section */
            oss.str("");
            oss << "index:" << std::setfill('0') << std::setw(5) << index;
            itk::EncapsulateMetaData(output->GetMetaDataDictionary(), oss.str(),
                        label);
            fprintf(stderr, "%s -> %i\n", oss.str().c_str(), label);
            fprintf(stderr, "%lu Voxels\n", stats->GetCount(label));
            
            /* Fill in the data in the output */
            result = (stats->GetMean(label)-totalstats->GetMean(label))/globalmean;
//            if(isnan(result) || isinf(result) || index == 0) {
//                fprintf(stderr, "*%f %f %li %f %f\n", result, 
//                        totalstats->GetSum(label), totalstats->GetCount(label),
//                        stats->GetMean(label), totalstats->GetMaximum(label));
//            }
            fullindex[SECTIONDIM] = index;
            output->SetPixel( fullindex, result);
            index++;
        }
    }
    
    /* For the rest of the timesteps the index/sections are already mapped */
    for(size_t ii = 1 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        stats->SetInput(extract(fmri_img, ii));
        stats->Update();
            
        fullindex[TIMEDIM] = ii;

        /* Write out the normalized value of each label */
        for(std::list<LabelType>::iterator it=rlabels.begin() ; 
                        it!=rlabels.end(); it++) {
            label = *it;
            if(label != 0) {
                oss.str("");
                oss << "section:" << std::setfill('0') << std::setw(5) << label;
                itk::ExposeMetaData(output->GetMetaDataDictionary(), oss.str(),
                            index);
//                fprintf(stderr, "%s -> %i\n", oss.str().c_str(), index);
            
                /* Fill in the data in the output */
                result = (stats->GetMean(label)-totalstats->GetMean(label))/globalmean;
//                if(isnan(result) || isinf(result)) {
//                    fprintf(stderr, "%f %f\n", totalstats->GetMean(label), 
//                                stats->GetMean(label));
//                }
                fullindex[SECTIONDIM] = index;
//                if(isnan(result) || isinf(result) || index == 0) {
//                    fprintf(stderr, "*%f %f %li %f %f\n", result, 
//                            totalstats->GetSum(label), totalstats->GetCount(label),
//                            stats->GetMean(label), totalstats->GetMaximum(label));
//                }
                output->SetPixel(fullindex, result);
            }
        }
        
    }

    return output;
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
//        tmp_reader->GetOutput()->SetMetaDataDictionary(
//                    dicomIO->GetMetaDataDictionary());
//    
//        itk::MetaDataDictionary& dict = 
//                    tmp_reader->GetOutput()->GetMetaDataDictionary();
        
        std::string value;
        dicomIO->GetValueFromTag("0020|0100", value);
        printf("Temporal Number: %s\n", value.c_str());
        
        reader[atoi(value.c_str())-1] = tmp_reader;

        dicomIO->GetValueFromTag("0018|0080", value);
        temporalres = atof(value.c_str())/1000.;
//        itk::EncapsulateMetaData(dict, "TemporalResolution", temporalres);
    
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
    unsigned int offset = 0;
    for(unsigned int ii=0; ii<seriesUID.size(); ii++) {
        if(ii*temporalres >= skip) {
            elevateFilter->PushBackInput(reader[ii]->GetOutput());
        } else {
            fprintf(stderr, "Skipping: %i, %f\n", ii, ii*temporalres);
            offset = (ii+1);
        }
    }
        

    //now elevateFilter can be used just like any other reader
    elevateFilter->Update();
    
    //create image readers
    Image4DType::Pointer fmri_img = elevateFilter->GetOutput();
//    fmri_img->CopyInformation(reader[0]->GetOutput());
    itk::EncapsulateMetaData(fmri_img->GetMetaDataDictionary(), "offset", offset);
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

