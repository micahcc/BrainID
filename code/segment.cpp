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

int detrend_avg(const Image4DType::Pointer fmri_img, Image4DType::IndexType index, 
            int knots, Image4DType::Pointer output)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, knots);

//    for(int i = 0 ; i < 4 ; i++) {
//        fprintf(stderr, "%zu ", index[i]);
//    }
//    fprintf(stderr, "\n");

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

//    for(int i = 0 ; i  < knots ; i++) {
//        fprintf(stderr, "%i Average: %f Count: %i, pos: %f\n", i, averages[i], 
//                    counts[i], xpos[i]);
//    }
//    for(int i = 0 ; i  < (knots-2) ; i++) {
//        fprintf(stderr, "%i Starts: %i\n", i, starts[i]);
//    }
    return 0;
};

int detrend_avg(const Image4DType::Pointer fmri_img, Image4DType::IndexType index, 
            const std::vector< unsigned int >& knots, 
            Image4DType::Pointer output)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, knots.size());

    double xpos[knots.size()];
    double level[knots.size()];

    for(unsigned int ii = 0 ; ii < knots.size() ; ii++) {
        xpos[ii] = knots[ii];
        level[ii] = 0;
        for(unsigned int jj = -1 ; jj <= 1 ; jj++) {
            index[3] = knots[ii];
            level[ii] += fmri_img->GetPixel(index);
        }
        level[ii] /= 3.;
    }

//    for(unsigned int ii = 0 ; ii < knots.size() ; ii++){
//        fprintf(stderr, "%f => %f\n", xpos[ii], level[ii]);
//    }

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


void getknots(std::vector<double>& knots, unsigned int num, 
            std::vector<Activation>& stim)
{
    std::list<Activation> fifo;
    std::list<Activation> mins;
    std::vector<Activation>::iterator it = stim.begin();
    double frame_begin = it->time;
    double sum_prev_prev = 1;
    double sum_prev = 0;
    double sum = 0;
    double time_prev = it->time;
    while(it->time - frame_begin < 10) {
        sum += ((it+1)->time-it->time)*(it->level);
        fifo.push_front(*it);
        it++;
    }
    sum -= (it-1)->level*(it->time-10);

    Activation tmp;

    while(fifo.size() != 1 || it != stim.end()) {
        printf("Sums: %f %f %f\n", sum_prev_prev, sum_prev, sum);
        for(std::list<Activation>::iterator tmpit = fifo.begin(); 
                    tmpit != fifo.end(); tmpit++) {
            printf("\t%f, %f\n", tmpit->time, tmpit->level);
        }

        if(sum > sum_prev && sum_prev < sum_prev_prev) {
            tmp.time = time_prev;
            tmp.level = sum_prev;
            mins.push_back(tmp);
        }

        sum_prev_prev = sum_prev;
        sum_prev = sum;
        time_prev = fifo.back().time+10;

        std::list<Activation>::reverse_iterator sec = ++fifo.rbegin();
        if(it != stim.end() && it->time - fifo.back().time - 10 < 
                    sec->time - fifo.back().time) {
            double delta = it->time - fifo.back().time - 10;
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

    //sort to shake out the lowest level areas
    mins.sort(compare_l);
    
    //debug
    fprintf(stderr, "Result:\n");
    for(std::list<Activation>::iterator tmpit = mins.begin(); 
            tmpit != mins.end(); tmpit++) {
        printf("\t%f, %f\n", tmpit->time, tmpit->level);
    }
    
    //find the first element to remove, then remove all elements beyond num
    std::list<Activation>::iterator tmpit = mins.begin();
    for(unsigned int i = 0 ; i <= num ;i++) {
        tmpit++;
    }
    mins.erase(tmpit, mins.end());

    //sort by time, then place into knots vector
    mins.sort(compare_t);
    knots.resize(num);
    tmpit = mins.begin();
    for(unsigned int i = 0 ; i < knots.size() ;i++) {
        knots[i] = tmpit->time;
        tmpit++;
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
    
    for(fmri_it.GoToBegin(); fmri_it != fmri_stop ; fmri_it.NextLine()) {
        for( ; !fmri_it.IsAtEndOfLine(); ++fmri_it) {
            detrend_avg(fmri_img, fmri_it.GetIndex(), knots, outimage);
        }
    }

    outimage->CopyInformation(fmri_img);
    outimage->SetMetaDataDictionary(fmri_img->GetMetaDataDictionary());
    return outimage;
}

Image4DType::Pointer normalizeByVoxel(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer mask, int nknots)
{
    fprintf(stderr, "Warning normalizeByVoxel is deprecated\n");
    std::vector<unsigned int> knots(nknots);
    unsigned int size = fmri_img->GetRequestedRegion().GetSize()[3];
    for(unsigned int i = 0 ; i < size; i++) {
        knots[i] = i*nknots/size;
    }

    Image4DType::Pointer spline = getspline(fmri_img, knots);
    
    {
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(spline);
    writer->SetFileName("spline.nii.gz");
    writer->Update();
    }

    Image3DType::Pointer average = get_average(fmri_img);
    double globalmean = get_average(fmri_img, mask);
    /* Rescale (fmri - spline)/avg*/

    SubF4::Pointer sub = SubF4::New();   
    DivCF::Pointer div = DivCF::New();
    sub->SetInput1(fmri_img);
    sub->SetInput2(spline);
    div->SetInput(sub->GetOutput());
    div->SetConstant(globalmean);
    div->Update();
    
    /* Add 1.5 sigma */
    SqrtF3::Pointer sqrt = SqrtF3::New();
    ScaleF::Pointer scale = ScaleF::New();
    AddF4::Pointer add = AddF4::New();
    
    Image3DType::Pointer variance = get_variance(div->GetOutput());
    {
    itk::ImageFileWriter< Image3DType >::Pointer writer = 
                itk::ImageFileWriter< Image3DType >::New();
    writer->SetInput(variance);
    writer->SetFileName("variance.nii.gz");
    writer->Update();
    }
    
    sqrt->SetInput(variance);
    scale->SetConstant(1.5);
    scale->SetInput(sqrt->GetOutput());
    scale->Update();

    Image4DType::Pointer sigma1_5 = stretch<double>(scale->GetOutput(), 
                fmri_img->GetRequestedRegion().GetSize()[3]);
    add->SetInput1(sigma1_5);
    add->SetInput2(div->GetOutput());
    add->Update();
    
    add->GetOutput()->CopyInformation(fmri_img);
    add->GetOutput()->SetMetaDataDictionary(
                fmri_img->GetMetaDataDictionary() );
    return add->GetOutput();
}

Image3DType::Pointer get_variance(const Image4DType::Pointer fmri_img)
{
    Image3DType::Pointer average = get_average(fmri_img);

    /* Used to zero out the addfilter */
    Image3DType::Pointer zero = extract(fmri_img, 0);
    zero->FillBuffer(0);
    
    /* Initialize the Addition */
    AddF3::Pointer add = AddF3::New();
    add->GraftOutput(zero);
    add->SetInput2(add->GetOutput());
    
    /* Initialize Subtraction */
    SubF3::Pointer sub = SubF3::New();   
    
    /* Initialize Subtraction */
    SqrF3::Pointer sqr = SqrF3::New();   
    
    /* Calculate Sum of Images */
    //SUM( (X_i - mu)^2 )
    for(size_t ii = 0 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        sub->SetInput1(extract(fmri_img, ii));
        sub->SetInput2(average);
        sqr->SetInput(sub->GetOutput());
        add->SetInput1(sqr->GetOutput());
        add->Update();
    }

    /* Calculate Average of Images */
    ScaleF::Pointer scale = ScaleF::New();
    scale->SetInput(add->GetOutput());
    scale->SetConstant(1./fmri_img->GetRequestedRegion().GetSize()[3]);
    scale->Update();
    return scale->GetOutput();
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
            std::list<int>& voxels)
{
    std::ostringstream oss;
    std::list<Image4DType::IndexType> positions;
    Image4DType::IndexType pos;
    for(std::list<int>::iterator it = voxels.begin() ; 
                    it != voxels.end();) {
        pos[0] = *it;
        it++;
        pos[1] = *it;
        it++;
        pos[2] = *it;
        it++;
        pos[3] = 0;
        positions.push_back(pos);
    }

    //initialize output
    Image4DType::Pointer output = initTimeSeries(fmri_img, positions.size());

    Image4DType::IndexType inindex;
    Image4DType::IndexType outindex = {{0, 0, 0, 0}};
    
    for(size_t ii = 0 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
        /* Write out the normalized value of each label */
        int jj = 0;
        for(std::list<Image4DType::IndexType>::iterator it = positions.begin() ; 
                        it != positions.end(); it++) {
            inindex = *it;
            inindex[TIMEDIM] = ii;

            outindex[0] = jj;
            outindex[TIMEDIM] = ii;
            
            /* Fill in the data in the output */
            fprintf(stderr, "%lu %lu %lu %lu -> %lu %lu %lu %lu\n", inindex[0], inindex[1],
                        inindex[2], inindex[3], outindex[0], outindex[1], 
                        outindex[2], outindex[3]);
            output->SetPixel(outindex, fmri_img->GetPixel(inindex));
            jj++;
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

Image4DType::Pointer pruneFMRI(const Image4DType::Pointer fmri_img,
            std::vector<Activation>& stim, double dt,
            unsigned int remove)
{
    /* Remove first few elements... */
    // .... from fmri_img
    Image4DType::Pointer new_img = prune<double>(fmri_img, 3, remove, 
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

/* Uses knots at points with at least a 15 second break */
Image4DType::Pointer deSpline(const Image4DType::Pointer fmri_img,
            unsigned int numknots, std::vector<Activation>& stim, double dt)
{
    /* Find good knots for spline */
    std::vector<double> knots_t;
    getknots(knots_t, numknots, stim);
    
    std::vector<unsigned int> knots_i(knots_t.size());
    for(unsigned int i = 0 ; i < knots_t.size() ; i++) 
        knots_i[i] = (unsigned int)knots_t[i]/dt;
        
    Image4DType::Pointer spline = getspline(fmri_img, knots_i);
    
    {
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
                itk::ImageFileWriter< Image4DType >::New();
    writer->SetInput(spline);
    writer->SetFileName("spline.nii.gz");
    writer->Update();
    }

    /* Rescale (fmri - spline)/avg*/
    
    Image4DType::Pointer avg = extrude(Tmean(fmri_img), 
                fmri_img->GetRequestedRegion().GetSize()[3]);
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
