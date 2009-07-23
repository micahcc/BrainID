#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <itkImage.h>
#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include <vector>
#include <iostream>
#include <iomanip>

#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>

/* Typedefs */
#define TIMEDIM 3
#define SERIESDIM 0

void outputVector(std::ostream& out, indii::ml::aux::vector mat) 
{
  unsigned int i;
  for (i = 0; i < mat.size(); i++) {
      out << std::setw(15) << mat(i);
  }
};

void outputMatrix(std::ostream& out, indii::ml::aux::matrix mat) 
{
  unsigned int i, j;
  for (j = 0; j < mat.size2(); j++) {
    for (i = 0; i < mat.size1(); i++) {
      out << std::setw(15) << mat(i,j);
    }
    out << std::endl;
  }
};


//write a vector to a dimension of an image
template <class T>
void writeVector(typename itk::OrientedImage< T, 4 >::Pointer out, int dir, 
            const indii::ml::aux::vector& input, 
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

    assert(i==input.size() && it.IsAtEndOfLine());
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
template <class T>
int readVector(const typename itk::OrientedImage< T, 4 >::Pointer in, int dir, 
            indii::ml::aux::vector& input, 
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
            typename itk::OrientedImage< T, 4 >::Pointer in,
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
