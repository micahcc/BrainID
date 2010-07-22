#ifndef TOOLS_H
#define TOOLS_H

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <itkImage.h>
#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "vnl/algo/vnl_fft_1d.h"
#include <itkCastImageFilter.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkMultiplyByConstantImageFilter.h>

#include "types.h"

void outputVector(std::ostream& out, indii::ml::aux::vector mat);
void outputMatrix(std::ostream& out, indii::ml::aux::matrix mat);
Image4DType::Pointer fft_image(Image4DType::Pointer inimg);
std::vector<Activation> read_activations(std::string base, std::string extra = "");
Image4DType::Pointer conditionFMRI(const Image4DType::Pointer fmri_img,
            int knots, std::vector<Activation>& stim);
Image3DType::Pointer Tmean(const Image4DType::Pointer fmri_img);
Image3DType::Pointer Tvar(const Image4DType::Pointer fmri_img);
//Image3DType::Pointer Tvar(const Image4DType::Pointer fmri_img);
Image4DType::Pointer extrude(const Image3DType::Pointer input, unsigned int len);
Image3DType::Pointer mse(const Image4DType::Pointer input1,
            const Image4DType::Pointer input2);
Image4DType::Pointer pctDiffOrient(const Image4DType::Pointer input1,
            const Image4DType::Pointer input2);
Image4DType::Pointer pctDiff(const Image4DType::Pointer input1,
            const Image4DType::Pointer input2);
Image3DType::Pointer get_rms(Image4DType::Pointer in);
Image4DType::Pointer extract_timeseries(Image4DType::Pointer in, 
            std::list< Image4DType::IndexType >& points);
itk::Image<DataType, 1>::Pointer getCanonical(std::vector<Activation> stim, double TR,
            double start, double stop);
Image3DType::Pointer median_absolute_deviation(const Image4DType::Pointer fmri_img);
Image3DType::Pointer mutual_info(uint32_t bins1, uint32_t bins2,
            const Image4DType::Pointer input1, const Image4DType::Pointer input2);


//dir1 should be the direction of several separate series
//dir2 should be the direction that you want to get rms of
//RMS for a non-zero mean signal is 
//sqrt(mu^2+sigma^2)
template<class Vector>
void get_rms(Image4DType::Pointer in, size_t dir1, size_t dir2, 
            Vector& out)
{
    for(unsigned int i = 0 ; i < out.size() ; i++) {
        out[i] = 0;
    }
    
    itk::ImageSliceIteratorWithIndex< Image4DType > 
                iter(in, in->GetRequestedRegion());
    iter.SetFirstDirection(dir1);
    iter.SetSecondDirection(dir2);
    iter.GoToBegin();

    int numelements = in->GetRequestedRegion().GetSize()[dir2];

    std::vector<double> mean(out.size(), 0);
    /* get average */
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        while(!iter.IsAtEndOfLine()) {
            mean[ii] += iter.Get();
            ii++;
            ++iter;
        }
        iter.NextLine();
    }
    for(size_t ii= 0 ; ii<mean.size() ; ii++) {
        mean[ii] /= numelements;
    }

    /* variance */
    iter.GoToBegin();
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        while(!iter.IsAtEndOfLine()) {
            out[ii] += pow(iter.Get()-mean[ii],2);
            ii++;
            ++iter;
        }
        iter.NextLine();
    }
    for(size_t ii= 0 ; ii<out.size() ; ii++)
        out[ii] /= numelements;


    //sqrt(mu^2 + var)
    for(size_t ii = 0 ; ii < out.size() ; ii++) {
        out[ii] = sqrt(pow(mean[ii], 2) + out[ii]);
    }
};


//write a vector to a dimension of an image
template <class T, class Vector>
int writeVector(typename itk::OrientedImage< T, 4 >::Pointer out, int dir, 
            const Vector& input, 
            typename itk::OrientedImage< T, 4 >::IndexType start)
{
    itk::ImageLinearIteratorWithIndex< itk::OrientedImage< T, 4 > >
                it(out, out->GetRequestedRegion());
    it.SetDirection(dir);
    it.SetIndex(start);

    size_t i;
    for(i = 0 ; i < input.size() && !it.IsAtEndOfLine() ; i++) {
        it.Set(input[i]);
        ++it;
    }
    return i;
};

//dir1 should be the first matrix dimension, dir2 the second
template <class T>
void writeMatrix(typename itk::OrientedImage< T, 4 >::Pointer out, int dir1, 
            int dir2, const indii::ml::aux::matrix& input, 
            typename itk::OrientedImage< T, 4 >::IndexType start)
{
    itk::ImageSliceIteratorWithIndex< itk::OrientedImage< T, 4 > >
                it(out, out->GetRequestedRegion());
    it.SetFirstDirection(dir1);
    it.SetSecondDirection(dir2);
    
    it.SetIndex(start);

    for(size_t j = 0 ; j < input.size2() && !it.IsAtEndOfSlice() ; j++) {
        for(size_t i = 0 ; i < input.size1() && !it.IsAtEndOfLine() ; i++) {
            it.Set(input(i,j));
        }
        it.NextLine();
    }
};


//read dimension of image into a vector
template <class T, class Vector>
int readVector(const typename itk::OrientedImage< T, 4 >::Pointer in, int dir, 
            Vector& input, 
            typename itk::OrientedImage< T, 4 >::IndexType start)
{
    itk::ImageLinearConstIteratorWithIndex<itk::OrientedImage< T, 4 > >
                it(in, in->GetRequestedRegion());
    it.SetDirection(dir);
    it.SetIndex(start);

    if((unsigned int)start[0] >= in->GetRequestedRegion().GetSize()[0] || 
                (unsigned int)start[1] >= in->GetRequestedRegion().GetSize()[1] || 
                (unsigned int)start[2] >= in->GetRequestedRegion().GetSize()[2] || 
                (unsigned int)start[3] >= in->GetRequestedRegion().GetSize()[3]) {
        return -1;
    }

    size_t i;
    for(i = 0 ; i < input.size() && !it.IsAtEndOfLine() ; i++, ++it) {
        input[i] = it.Get();
    }

    if((unsigned int)start[TIMEDIM]+1 >= 
                in->GetRequestedRegion().GetSize()[TIMEDIM]) {
        return 1;
    } else {
        return 0;
    }
};

template <class T>
typename itk::OrientedImage< T, 4 >::Pointer concat(
            std::vector< typename itk::OrientedImage< T, 4 >::Pointer >& in,
            int dir )
{
    typedef itk::OrientedImage< T, 4>  ImageType;
    int outsize = 0;

    
    /* Initialize Iterators */
    std::vector< itk::ImageLinearConstIteratorWithIndex< ImageType > > iterators;
    for(size_t ii = 0 ; ii < in.size() ; ii++) {
        for(int jj = 0 ; jj < 4 ; jj++) {
            if(jj != dir && in[ii]->GetRequestedRegion().GetSize()[jj] != 
                        in[0]->GetRequestedRegion().GetSize()[jj]) {
                return NULL;
            }
        }
        
        iterators.push_back( itk::ImageSliceIteratorWithIndex< ImageType >( 
                in[ii], in[ii]->GetRequestedRegion()) );
        iterators[ii].SetDirection(dir);
        iterators[ii].GoToBegin();
        outsize += in[ii]->GetRequestedRegion().GetSize()[dir];
    }
    
    typename ImageType::RegionType out_region;
    typename ImageType::IndexType out_index = {{ 0, 0, 0, 0 }};
    typename ImageType::SizeType out_size = 
                in[0]->GetRequestedRegion().GetSize();
    out_size[dir] = outsize;
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    typename ImageType::Pointer newout = 
                ImageType::New();
    newout->SetRegions( out_region );
    newout->Allocate();
    
    typename itk::ImageLinearIteratorWithIndex<ImageType>
                newout_it(newout, newout->GetRequestedRegion());
    newout_it.SetDirection(dir);
    newout_it.GoToBegin();
    
    while(!newout_it.IsAtEnd()) {
        for(size_t ii=0 ; ii<in.size() ; ii++) {
            while(!iterators[ii].IsAtEndOfLine()) {
                newout_it.Set(iterators[ii].Get());
                ++iterators[ii];
                ++newout_it;
            }
            iterators[ii].NextLine();
        }

        if(!newout_it.IsAtEndOfLine())
            return NULL;

        newout_it.NextLine();
    }

    newout->CopyInformation(in[0]);
    newout->SetMetaDataDictionary(in[0]->GetMetaDataDictionary());

    return newout;
};

/* Removes all but the indices between start and stop, inclusive */
template <class T>
typename itk::OrientedImage< T, 4 >::Pointer prune(
            const typename itk::OrientedImage< T, 4 >::Pointer in,
            int dir, int start, int stop)
{
    typedef itk::OrientedImage< T, 4>  ImageType;
    
    /* Check sizes */
    if(stop >= (int)in->GetRequestedRegion().GetSize()[dir]) 
        stop = (int)in->GetRequestedRegion().GetSize()[dir]-1;
    
    if(start < 0)
        start = 0;

    int outsize = stop-start+1;
    
    typename itk::ImageLinearIteratorWithIndex<ImageType>
                it(in, in->GetRequestedRegion());
    it.SetDirection(dir);
    it.GoToBegin();
    
    typename ImageType::RegionType out_region;
    typename ImageType::IndexType out_index = {{ 0, 0, 0, 0 }};
    typename ImageType::SizeType out_size = in->GetRequestedRegion().GetSize();
    out_size[dir] = outsize;
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    typename ImageType::Pointer newout = 
                ImageType::New();
    newout->SetRegions( out_region );
    newout->Allocate();
    
    typename itk::ImageLinearIteratorWithIndex<ImageType>
                newout_it(newout, newout->GetRequestedRegion());
    newout_it.SetDirection(dir);
    newout_it.GoToBegin();
    
    while(!newout_it.IsAtEnd()) {
        while(!newout_it.IsAtEndOfLine()) {
            out_index = newout_it.GetIndex();
            out_index[dir] = out_index[dir] + start;
            it.SetIndex(out_index);
            newout_it.Set(it.Get());
            ++newout_it;
        }
        newout_it.NextLine();
    }

    newout->CopyInformation(in);
    newout->SetMetaDataDictionary(in->GetMetaDataDictionary());

    return newout;
};

template<class T, unsigned int SIZE1, class U, unsigned int SIZE2>
typename itk::OrientedImage<T, SIZE1>::Pointer applymask(
            typename itk::OrientedImage<T, SIZE1>::Pointer input, 
            typename itk::OrientedImage<U, SIZE2>::Pointer mask)
{
    /* Make Copy For Output */
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

template <class SrcType, class DstType >
void copyInformation(typename SrcType::Pointer src, typename DstType::Pointer dst)
{
    typename SrcType::PointType srcOrigin = src->GetOrigin();
    typename DstType::PointType dstOrigin = dst->GetOrigin();
    
    typename SrcType::DirectionType srcDir = src->GetDirection();
    typename DstType::DirectionType dstDir = dst->GetDirection();

    typename SrcType::SpacingType srcSpace = src->GetSpacing();
    typename DstType::SpacingType dstSpace = dst->GetSpacing();

    unsigned int mindim = src->GetImageDimension() < dst->GetImageDimension() ?
                src->GetImageDimension() : dst->GetImageDimension();
    unsigned int maxdim = src->GetImageDimension() > dst->GetImageDimension() ?
                src->GetImageDimension() : dst->GetImageDimension();

    for(unsigned int ii = 0 ; ii < mindim ; ii++) 
        dstOrigin[ii] = srcOrigin[ii];
    for(unsigned int ii = mindim ; ii < maxdim ; ii++) 
        dstOrigin[ii] = 0;
    
    for(unsigned int ii = 0 ; ii < mindim ; ii++) 
        dstSpace[ii] = srcSpace[ii];
    for(unsigned int ii = mindim ; ii < maxdim ; ii++) 
        dstSpace[ii] = 1;
    
    for(unsigned int ii = 0 ; ii < maxdim ; ii++) {
        for(unsigned int jj = 0 ; jj < maxdim ; jj++) {
            if(ii < mindim && jj < mindim) 
                dstDir(ii,jj) = srcDir(ii,jj);
            else if(ii == jj)
                dstDir(ii,jj) = 1;
            else 
                dstDir(ii,jj) = 0;
        }
    }

    dst->SetSpacing(dstSpace);
    dst->SetDirection(dstDir);
    dst->SetOrigin(dstOrigin);
}

#endif// TOOLS_H
