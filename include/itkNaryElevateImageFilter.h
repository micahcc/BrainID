/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkNaryElevateImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2007/02/05 10:22:00 $
  Version:   $Revision: 1.0 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkNaryElevateImageFilter_h
#define __itkNaryElevateImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkArray.h"

namespace itk
{

namespace Functor
{

// this functor is used if the corresponding template function is left undefined
template <class TInput, class TOutput>
class DefaultElevateFunction
{
public:

  DefaultElevateFunction() {}
  ~DefaultElevateFunction() {}

  inline void operator()(const Array<TInput> &A,const Array<double> &T,Array<TOutput> &R)
  {
    int size=A.size();
    for (int i=0; i<size; i++) R[i]=static_cast<TOutput>(A[i]); // just copy without change
  }
};

}

/** \class NaryElevateImageFilter
 * \brief Implements generic elevation of a time series of several nD inputs to a single n+1D output.
 *
 * Contributed by Stefan Roettger, Siemens Medical MR, Mar. 2007
 *
 * This class takes a time series of inputs with the same dimension and type and
 * generates an output image of the next higher dimension which contains the entire time series.
 * For example, a set of 3D inputs will be composed into a 4D output with
 * the highest dimension corresponding to the time axis.
 * Additionally, a functor style template parameter can be used for
 * per-pixel-wise manipulation of the time curves of the input data.
 * The functor can be ommited, then the data is just copied.
 * The filter also provides the min, the max, the mean and
 * an approximation of the noise level of the generated output.
 *
 * A typical times series produced in clinical practice is a MR breast angiography.
 * Gadolinium contrast agent is infused and T1-weighted 3D MR-images are taken every minute.
 * The procedure usually generates 5-10 time points which show the perfusion of the breast tissue.
 * The itkNaryElevateImageFilter takes the 3D breast images and constructs a single 4D image
 * containing the information of the entire scan.
 * After this, the itkUnaryRetractFilter can be used to calculate certain characteristics of
 * the time curves which give insight into the malignancy of the breast tissue.
 * Characteristic operations are MIPt (Maximum Intensity [Projection] over time) and
 * WI (Wash-In or temporal derivative), for example.
 * A tumor usually shows higher WI values than normal tissue, so that this parameter
 * (amongst a variety of others) can be used to diagnose breast cancer.
 * 
 * \ingroup IntensityImageFilters Multithreaded
 */

template <class TInputImage, class TOutputImage, class TFunction =
  typename Functor::DefaultElevateFunction<
    typename TInputImage::PixelType,
    typename TOutputImage::PixelType> >
 class ITK_EXPORT NaryElevateImageFilter:
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:

  /** Standard class typedefs. */
  typedef NaryElevateImageFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NaryElevateImageFilter, ImageToImageFilter);

  /** Some typedefs. */
  typedef TFunction FunctorType;
  typedef TInputImage InputImageType;
  typedef typename InputImageType::Pointer     InputImagePointer;
  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename InputImageType::PixelType   InputImagePixelType;
  typedef typename InputImageType::SizeType    InputImageSizeType;
  typedef typename InputImageType::IndexType   InputImageIndexType;
  typedef typename InputImageType::PointType   InputImageOriginType;
  typedef typename InputImageType::SpacingType InputImageSpacingType;
  typedef TOutputImage OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::SizeType    OutputImageSizeType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImageOriginType;
  typedef typename OutputImageType::SpacingType OutputImageSpacingType;
  typedef Array<InputImagePixelType> NaryInputType;
  typedef Array<OutputImagePixelType> NaryOutputType;

  /** ImageDimension constants. */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Get functions. */
  itkGetMacro(DataMin,float);
  itkGetMacro(DataMax,float);
  itkGetMacro(DataMean,float);
  itkGetMacro(NoiseThres,float);

  // get the actual functor
  FunctorType& GetFunctor() {return m_Functor;}

  // set the actual functor
  void SetFunctor(FunctorType& functor)
  {
    m_Functor = functor;
    this->Modified();
  }

protected:

  NaryElevateImageFilter();
  virtual ~NaryElevateImageFilter() {};

  void GenerateOutputInformation();
  void GenerateData();

private:

  NaryElevateImageFilter(const Self&); // intentionally not implemented
  void operator=(const Self&);         // intentionally not implemented

  float m_DataMin;
  float m_DataMax;
  float m_DataMean;
  float m_NoiseThres;

  FunctorType m_Functor;
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNaryElevateImageFilter.txx"
#endif

#endif
