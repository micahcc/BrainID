//image readers
#include "itkImageFileReader.h"

//standard libraries
#include <ctime>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <list>

#include <itkImageFileWriter.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkMetaDataDictionary.h>
#include <itkNaryElevateImageFilter.h>
#include <itkMaskImageFilter.h>

#include "segment.h"
#include "tools.h"
#include <itkLabelStatisticsImageFilterMod.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageLinearConstIteratorWithIndex.h>

#include <itkSubtractConstantFromImageFilter.h>
#include <itkDivideByConstantImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkSqrtImageFilter.h>
#include <itkNaryAddImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkDivideByConstantImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

#include <gsl/gsl_spline.h>

typedef itk::AddImageFilter< Image3DType > AddF3;
typedef itk::AddImageFilter< Image4DType > AddF4;

typedef itk::SubtractImageFilter< Image3DType > SubF3;
typedef itk::SubtractImageFilter< Image4DType > SubF4;

typedef itk::MultiplyByConstantImageFilter< Image3DType, double, Image3DType > ScaleF;
typedef itk::BinaryThresholdImageFilter<Label3DType, Label3DType> ThreshF;
typedef itk::LabelStatisticsImageFilterMod<Image3DType, Label3DType> StatF3D;
typedef itk::MaskImageFilter<Image3DType, Label3DType, Image3DType> MaskF;
//typedef itk::LabelStatisticsImageFilter<Image3DType, Label3DType> StatF3D;

typedef itk::LabelStatisticsImageFilterMod<Image4DType, Label4DType> StatF4D;
//typedef itk::LabelStatisticsImageFilter<Image4DType, Label4DType> StatF4D;
typedef itk::NaryAddImageFilter< Image3DType, Image3DType > AddNF;
typedef itk::SquareImageFilter< Image3DType, Image3DType > SqrF3;
typedef itk::SqrtImageFilter< Image3DType, Image3DType > SqrtF3;
typedef itk::SubtractConstantFromImageFilter< Image4DType, double, 
            Image4DType > SubCF;
typedef itk::DivideImageFilter< Image4DType, Image4DType, Image4DType > DivF;
typedef itk::DivideByConstantImageFilter< Image4DType, double, Image4DType > DivCF;

/* Remove any elements that arent' in the reference list 
 * ref will be sorted, but otherwise will be unchanged*/

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

/* Returns:
 * -1 if before is the lowest
 *  0 if current is the lowest or current ties for lowest
 *  1 if after is the lowest 
 */
int min(int before, int current, int after) {
    if(before < current) {
        if(before < after) {
            return -1;
        } else { //before >= after 
            return 1;
        }
    } else { //before >= current
        if( current <= after) {
            return 0;
        } else { //current > after 
            return 1;
        }
    }
}


Image3DType::Pointer get_average(const Image4DType::Pointer fmri_img)
{
    /* Used to zero out the addfilter */
    Image3DType::Pointer zero = extract(fmri_img, 0);
    zero->FillBuffer(0);
    
    /* Initialize the Addition */
    AddF3::Pointer add = AddF3::New();
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
    
    if(stats->GetNumberOfLabels() == 2)
        return stats->GetMean(1);
    else 
        return stats->GetMean(0);
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
        
    itk::GDCMImageIO::Pointer dicomIO;
    std::string value;

    double temporalres = 2;
    //reorder the input based on temporal number.
    while (seriesItr!=seriesEnd)
    {
        itk::ImageSeriesReader<Image3DType>::Pointer tmp_reader = 
                    itk::ImageSeriesReader<Image3DType>::New();

        dicomIO = itk::GDCMImageIO::New();
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
        
    const std::vector<std::string>& keys = dicomIO->GetMetaDataDictionary().GetKeys();
    std::string label;
    for(size_t i = 0 ; i < keys.size() ;i++) {
        if(dicomIO->GetValueFromTag(keys[i], value)) {
            fprintf(stdout, "%s", keys[i].c_str());
            if(dicomIO->GetLabelFromTag(keys[i], label)){
                fprintf(stdout, " -> %s: %s\n", label.c_str(), value.c_str());
                itk::EncapsulateMetaData(fmri_img->GetMetaDataDictionary(), label, value);
            } else {
                fprintf(stdout, ": %s\n", value.c_str());
                itk::EncapsulateMetaData(fmri_img->GetMetaDataDictionary(), keys[i], value);
            }
        }
    }
   

    delete[] reader;
    return fmri_img;
};

Image4DType::Pointer pruneFMRI(const Image4DType::Pointer fmri_img,
            std::vector<Activation>& stim, double dt,
            unsigned int remove)
{
    /* Remove first few elements... */
    // .... from fmri_img
    Image4DType::Pointer new_img = prune<float>(fmri_img, 3, remove, 
                fmri_img->GetRequestedRegion().GetSize()[3]);

    // .... from stimulus, then shift times
    std::vector<Activation>::iterator it = stim.begin();
    std::vector<Activation>::iterator start = stim.begin();
    while(it != stim.end()) {
        if(it->time < dt*remove) {
            start = it;
        } else {
            break;
        }
        it++;
    }
    start->time = dt*remove;
    stim.erase(stim.begin(), start);
    
    for(unsigned int i = 0 ; i < stim.size() ;i++) {
        stim[i].time -= dt*remove;
    }

    return new_img;
}

/**************************************************************************
 * Blind Spline Generation
**************************************************************************/
int detrend_lmin(const Image4DType::Pointer fmri_img, Image4DType::IndexType index, 
            int ties, Image4DType::Pointer output)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, ties);

    /* Go to index and start at time 0 at that voxel*/
    itk::ImageLinearConstIteratorWithIndex< Image4DType > 
                fmri_it(fmri_img, fmri_img->GetRequestedRegion());
    fmri_it.SetDirection(3);
    fmri_it.SetIndex(index);

    Image4DType::IndexType beforeindex = index;
    Image4DType::IndexType afterindex = index;
    Image4DType::IndexType tmpindex = index;

    double levels[ties];
    double xpos[ties];
    int positions[ties];

    int length = fmri_img->GetRequestedRegion().GetSize()[3];
    positions[0] = 0;
    positions[ties-1] = length-1;
    for(int i = 1 ; i < ties-1 ; i++) {
        positions[i] = positions[i-1] + (length - positions[i-1])/(ties-i);
    }

    for(int i = 0 ; i < ties ; i++) {
        int dir = 0;
        do{
            beforeindex[3] = (positions[i] == 0) ? positions[i] : positions[i]-1;
            afterindex[3] = (positions[i] == length-1) ? positions[i] : positions[i]+1;
            tmpindex[3] = positions[i];

            dir = min(fmri_img->GetPixel(beforeindex), fmri_img->GetPixel(tmpindex),
                        fmri_img->GetPixel(afterindex));
            positions[i] += dir;
        } while( dir != 0 );
    }
    
    for(int i = 0 ; i < ties ; i++) {
        tmpindex[3] = positions[i];
        levels[i] = fmri_img->GetPixel(tmpindex);
        xpos[i] = positions[i];
    }

    itk::ImageLinearIteratorWithIndex< Image4DType > 
                out_it(output, output->GetRequestedRegion());
    out_it.SetDirection(3);
    out_it.SetIndex(index);
    gsl_spline_init(spline, xpos, levels, ties);
    for(out_it.GoToBeginOfLine(); !out_it.IsAtEndOfLine(); ++out_it) {
        out_it.Set(gsl_spline_eval(spline, out_it.GetIndex()[3], acc));
    }

    return 0;
}

int detrend_median(const Image4DType::Pointer fmri_img, Image4DType::IndexType index, 
            int knots, Image4DType::Pointer output)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, knots);

    /* Go to index and start at time 0 at that voxel*/
    itk::ImageLinearConstIteratorWithIndex< Image4DType > 
                fmri_it(fmri_img, fmri_img->GetRequestedRegion());
    fmri_it.SetDirection(3);
    fmri_it.SetIndex(index);

    double xpos[knots];
    double medians[knots];

    int length = fmri_img->GetRequestedRegion().GetSize()[3];
    double rsize = (double)length / (knots-1);

    std::vector< std::list<double> > points(knots);
    
    for(fmri_it.GoToBeginOfLine(); !fmri_it.IsAtEndOfLine(); ++fmri_it) {
        unsigned int i = fmri_it.GetIndex()[3];
        if(i < rsize/2) {
//            printf("%u -> 0\n", i);
            points.front().push_back(fmri_it.Get());
        } else if(i > length - rsize/2) {
//            printf("%u -> last (%zu)\n", i, points.size());
            points.back().push_back(fmri_it.Get());
        } else {
//            printf("%u -> %i\n", i, 1+(int)(i/rsize-1./2));
            points[1+(int)(i/rsize-1./2)].push_back(fmri_it.Get());
        }
    }

    for(unsigned int i = 0 ; i < points.size() ; i++) {
        points[i].sort();
        if(points[i].size()%2 == 0) {
            std::list<double>::iterator it = points[i].begin();
            for(unsigned int j = 0 ; j != points[i].size()/2-1 ; j++)
                it++;
            medians[i] = *it;
            it++;
            medians[i] += *it;
            medians[i] /= 2.;
        } else {
            std::list<double>::iterator it = points[i].begin();
            for(unsigned int j = 0 ; j != points[i].size()/2 ; j++)
                it++;
            medians[i] = *it;
        }

//        printf("Median of: ");
//        std::list<double>::iterator it = points[i].begin();
//        while(it != points[i].end()) {
//            printf("%f, ", *it);
//            it++;
//        }
//        printf("\nis %f\n", medians[i]);
    }

    xpos[0] = 0;
    xpos[knots-1] = length-1;

    for(int i = 1 ; i < knots-1; i++) {
        xpos[i] = rsize/2+(i-1)*rsize+rsize/2;
        printf("xpos %i: %f\n", i, xpos[i]);
    }
    
    itk::ImageLinearIteratorWithIndex< Image4DType > 
                out_it(output, output->GetRequestedRegion());
    out_it.SetDirection(3);
    out_it.SetIndex(index);
    gsl_spline_init(spline, xpos, medians, knots);
    for(out_it.GoToBeginOfLine(); !out_it.IsAtEndOfLine(); ++out_it) {
        out_it.Set(gsl_spline_eval(spline, out_it.GetIndex()[3], acc));
    }

    return 0;
};

int detrend_avg(const Image4DType::Pointer fmri_img, Image4DType::IndexType index, 
            int knots, Image4DType::Pointer output)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, knots);

    /* Go to index and start at time 0 at that voxel*/
    itk::ImageLinearConstIteratorWithIndex< Image4DType > 
                fmri_it(fmri_img, fmri_img->GetRequestedRegion());
    fmri_it.SetDirection(3);
    fmri_it.SetIndex(index);

    double averages[knots];
    int counts[knots];
    int starts[knots-2];

    int length = fmri_img->GetRequestedRegion().GetSize()[3];
    double rsize = length / (knots-2);
    for(int i = 0 ; i < (knots-2) ; i++) {
        starts[i] = rsize*i;
    }
    for(int i = 0 ; i < knots ; i++) {
        counts[i] = 0;
        averages[i] = 0;
    }
    
    int region = 0;
    for(fmri_it.GoToBeginOfLine(); !fmri_it.IsAtEndOfLine(); ++fmri_it) {
        /* Figure out Region */
        if(fmri_it.GetIndex()[3] < length/((knots-2)*2)) {
            averages[0] += fmri_it.Get();
            counts[0]++;
        } else if(fmri_it.GetIndex()[3] > (length - length/((knots-2)*2))) {
            averages[(knots-2)+1] += fmri_it.Get();
            counts[(knots-2)+1]++;
        }

        /* Check to see if the index is for the last region(since there is no)
         * start for the region after that */
        if(fmri_it.GetIndex()[3] > starts[(knots-2)-1]) {
            averages[(knots-2)] += fmri_it.Get();
            counts[(knots-2)]++;
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

    double xpos[knots];
    xpos[0] = 0;
    xpos[(knots-2)+1] = length-1;;
    xpos[(knots-2)] = (length + starts[(knots-2)-1])/2;
    for(int i = 0 ; i < (knots-2)-1 ; i++){
        xpos[i+1] = (starts[i+1] + starts[i])/2;
    }

    for(int i = 0 ; i < knots ; i++) {
        averages[i] /= counts[i];
    }

    itk::ImageLinearIteratorWithIndex< Image4DType > 
                out_it(output, output->GetRequestedRegion());
    out_it.SetDirection(3);
    out_it.SetIndex(index);
    gsl_spline_init(spline, xpos, averages, knots);
    for(out_it.GoToBeginOfLine(); !out_it.IsAtEndOfLine(); ++out_it) {
        out_it.Set(gsl_spline_eval(spline, out_it.GetIndex()[3], acc));
    }

    return 0;
};

Image4DType::Pointer getspline(const Image4DType::Pointer fmri_img, 
            unsigned int knots)
{
    Image4DType::Pointer outimage = Image4DType::New();
    outimage->SetRegions(fmri_img->GetRequestedRegion());
    outimage->Allocate();
    outimage->FillBuffer(0);

    /* Fmri Iterators */
    itk::ImageLinearConstIteratorWithIndex< Image4DType > 
                fmri_it(fmri_img, fmri_img->GetRequestedRegion());
    fmri_it.SetDirection(0);
    
    itk::ImageLinearConstIteratorWithIndex< Image4DType >
                fmri_stop(fmri_img, fmri_img->GetRequestedRegion());
    fmri_stop.SetDirection(3);
    fmri_stop.GoToBegin();
    ++fmri_stop;

    for(fmri_it.GoToBegin(); fmri_it != fmri_stop ; fmri_it.NextLine()) {
        for( ; !fmri_it.IsAtEndOfLine(); ++fmri_it) {
            detrend_avg(fmri_img, fmri_it.GetIndex(), knots, outimage);
        }
    }

    outimage->CopyInformation(fmri_img);
    outimage->SetMetaDataDictionary(fmri_img->GetMetaDataDictionary());
    return outimage;
}

Image4DType::Pointer getspline_m(const Image4DType::Pointer fmri_img, 
            unsigned int knots)
{
    Image4DType::Pointer outimage = Image4DType::New();
    outimage->SetRegions(fmri_img->GetRequestedRegion());
    outimage->Allocate();
    outimage->FillBuffer(0);

    /* Fmri Iterators */
    itk::ImageLinearConstIteratorWithIndex< Image4DType > 
                fmri_it(fmri_img, fmri_img->GetRequestedRegion());
    fmri_it.SetDirection(0);
    
    itk::ImageLinearConstIteratorWithIndex< Image4DType >
                fmri_stop(fmri_img, fmri_img->GetRequestedRegion());
    fmri_stop.SetDirection(3);
    fmri_stop.GoToBegin();
    ++fmri_stop;

    for(fmri_it.GoToBegin(); fmri_it != fmri_stop ; fmri_it.NextLine()) {
        for( ; !fmri_it.IsAtEndOfLine(); ++fmri_it) {
            detrend_median(fmri_img, fmri_it.GetIndex(), knots, outimage);
        }
    }

    outimage->CopyInformation(fmri_img);
    outimage->SetMetaDataDictionary(fmri_img->GetMetaDataDictionary());
    return outimage;
}

Image4DType::Pointer deSplineBlind(const Image4DType::Pointer fmri_img,
            unsigned int numknots)
{
    std::cerr << "Making Spline" << std::endl;
//    Image4DType::Pointer spline = getspline(fmri_img, numknots);
    Image4DType::Pointer spline = getspline_m(fmri_img, numknots);
    
    {
    printf("Writing spline\n");
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(spline);
    writer->SetFileName("spline.nii.gz");
    writer->Update();
    }

    /* Rescale (fmri - spline)/avg*/
    Image4DType::Pointer avg = extrude(Tmean(fmri_img), 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    {
    printf("Writing fmri_img\n");
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(fmri_img);
    writer->SetFileName("fmri_img.nii.gz");
    writer->Update();
    }
    {
    printf("Writing avg\n");
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(avg);
    writer->SetFileName("avg.nii.gz");
    writer->Update();
    }
    SubF4::Pointer sub = SubF4::New();   
    DivF::Pointer div = DivF::New();   
    sub->SetInput1(fmri_img);
    sub->SetInput2(spline);
    div->SetInput1(sub->GetOutput());
    div->SetInput2(avg);
    div->Update();

    div->GetOutput()->CopyInformation(fmri_img);
    div->GetOutput()->SetMetaDataDictionary(
                fmri_img->GetMetaDataDictionary() );
    return div->GetOutput();
}

/**************************************************************************
 * Informed Spline Generation
**************************************************************************/
int detrend_avg(const Image4DType::Pointer fmri_img, Image4DType::IndexType index, 
            const std::vector< unsigned int >& knots, 
            Image4DType::Pointer output)
{
    static gsl_interp_accel *acc = gsl_interp_accel_alloc();
    static gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, knots.size());

    double xpos[knots.size()];
    double level[knots.size()];
    for(unsigned int ii = 0 ; ii < knots.size() ; ii++) {
        xpos[ii] = knots[ii];
        level[ii] = 0;
        for(int jj = -1 ; jj <= 1; jj++) {
            index[3] = knots[ii]+jj;
            level[ii] += fmri_img->GetPixel(index);
        }
        level[ii] /= 3.;
        
    }

    itk::ImageLinearIteratorWithIndex< Image4DType > 
                out_it(output, output->GetRequestedRegion());
    out_it.SetDirection(3);
    out_it.SetIndex(index);
    gsl_spline_init(spline, xpos, level, knots.size());
    for(out_it.GoToBeginOfLine(); !out_it.IsAtEndOfLine(); ++out_it) {
        out_it.Set(gsl_spline_eval(spline, out_it.GetIndex()[3], acc));
    }

    return 0;
}

bool compare_l(Activation A, Activation B)
{
    return A.level < B.level;
}

bool compare_t(Activation A, Activation B)
{
    return A.time < B.time;
}


void getknots(std::list<Activation>& mins, 
            std::vector<Activation>& stim)
{
    const double RANGE = 20;
    std::list<Activation> fifo;
    std::vector<Activation>::iterator it = stim.begin();
    double frame_begin = it->time;
    double sum_prev_prev = 1;
    double sum_prev = 0;
    double sum = 0;
    double time_prev = RANGE;
    while(it->time - frame_begin < RANGE) {
        sum += ((it+1)->time-it->time)*(it->level);
        fifo.push_front(*it);
        it++;
    }
    sum -= (it-1)->level*(it->time-RANGE);

    Activation tmp;

    while(fifo.size() != 1 || it != stim.end()) {
        //find local minima
        if(sum > sum_prev && sum_prev <= sum_prev_prev) {
            tmp.time = time_prev;
            tmp.level = sum_prev;
            mins.push_back(tmp);
        }

        sum_prev_prev = sum_prev;
        sum_prev = sum;
        time_prev = fifo.back().time+RANGE;

        std::list<Activation>::reverse_iterator sec = ++fifo.rbegin();
        if(fifo.size() <= 1 || ( it != stim.end() && 
                    it->time - fifo.back().time - RANGE < 
                    sec->time - fifo.back().time ) ) {
            double delta = it->time - fifo.back().time - RANGE;
            sum -= fifo.back().level*delta;
            sum += delta*fifo.front().level;
            fifo.back().time += delta;
            fifo.push_front(*it);
            it++;
        } else {
            double delta = sec->time - fifo.back().time;
            sum -= fifo.back().level*delta;
            sum += delta*fifo.front().level;
            fifo.pop_back();
        }
    }
}

Image4DType::Pointer getspline(const Image4DType::Pointer fmri_img,
            const std::vector<unsigned int>& knots)
            
{
    Image4DType::Pointer outimage = Image4DType::New();
    outimage->SetRegions(fmri_img->GetRequestedRegion());
    outimage->Allocate();
    outimage->FillBuffer(0);

    /* Fmri Iterators */
    itk::ImageLinearConstIteratorWithIndex< Image4DType > 
                fmri_it(fmri_img, fmri_img->GetRequestedRegion());
    fmri_it.SetDirection(0);
    
    itk::ImageLinearConstIteratorWithIndex< Image4DType >
                fmri_stop(fmri_img, fmri_img->GetRequestedRegion());
    fmri_stop.SetDirection(3);
    fmri_stop.GoToBegin();
    ++fmri_stop;

    std::cerr << "Knots:" << std::endl;
    for(unsigned int ii = 0 ; ii < knots.size() ;ii++) {
        std::cerr << knots[ii] << std::endl;
    }
    
    for(fmri_it.GoToBegin(); fmri_it != fmri_stop ; fmri_it.NextLine()) {
        for( ; !fmri_it.IsAtEndOfLine(); ++fmri_it) {
            detrend_avg(fmri_img, fmri_it.GetIndex(), knots, outimage);
        }
    }

    outimage->CopyInformation(fmri_img);
    outimage->SetMetaDataDictionary(fmri_img->GetMetaDataDictionary());
    return outimage;
}


/* Uses knots at points with at least a 15 second break */
Image4DType::Pointer deSplineByStim(const Image4DType::Pointer fmri_img,
            unsigned int numknots, std::vector<Activation>& stim, double dt)
{
    /* Find good knots for spline */
    std::list<Activation> knots_a;
    std::list<Activation> knots_b;

    std::cerr << "Getting knots" << std::endl;
    getknots(knots_a, stim);
    
    std::cerr << "Picking good knots" << std::endl;
    //sort by l, then start adding elements to knots_b
    //before inserting a knot, make sure there are no knots
    //close in time, if there are, throw the knot out
    knots_a.sort(compare_l);

    for(std::list<Activation>::iterator it = knots_a.begin() ; 
                    it != knots_a.end() && knots_b.size() < numknots; it++) {
        //check that the point is valid
        if(it->time/dt >= fmri_img->GetRequestedRegion().GetSize()[3]-1 ||
                        it->time/dt <= 1) 
            continue;

        //check that the point isn't too close to any other point
        std::list<Activation>::iterator it2 = knots_b.begin();
        while(it2 != knots_b.end()) {
            if(it->time < it2->time + 8*dt && it->time > it2->time - 8*dt) {
                break;
            }
            it2++;
        }

        //no nearby points, go ahead and add
        if(it2 == knots_b.end()) 
            knots_b.push_back(*it);
    }

    knots_b.sort(compare_t);

    /* Change the times into positions in the vector, and use knots in the range */
    std::vector<unsigned int> knots_final(knots_b.size());
    std::list<Activation>::iterator itb = knots_b.begin();
    std::vector<unsigned int>::iterator itf = knots_final.begin();
    while(itb != knots_b.end()) {
        *itf = (unsigned int)itb->time/dt;
        itb++;
        itf++;
    }
    
    std::cerr << "Making Spline" << std::endl;
    Image4DType::Pointer spline = getspline(fmri_img, knots_final);
    
    {
    printf("Writing spline");
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(spline);
    writer->SetFileName("spline.nii.gz");
    writer->Update();
    }

    /* Rescale (fmri - spline)/avg*/
    
    
    Image4DType::Pointer avg = extrude(Tmean(fmri_img), 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    {
    printf("Writing fmri_img");
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(fmri_img);
    writer->SetFileName("fmri_img.nii.gz");
    writer->Update();
    }
    {
    printf("Writing avg");
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(avg);
    writer->SetFileName("avg.nii.gz");
    writer->Update();
    }
    SubF4::Pointer sub = SubF4::New();   
    DivF::Pointer div = DivF::New();   
    sub->SetInput1(fmri_img);
    sub->SetInput2(spline);
    div->SetInput1(sub->GetOutput());
    div->SetInput2(avg);
    div->Update();

    div->GetOutput()->CopyInformation(fmri_img);
    div->GetOutput()->SetMetaDataDictionary(
                fmri_img->GetMetaDataDictionary() );
    return div->GetOutput();
}
