#ifndef _itkNaryElevateImageFilter_txx
#define _itkNaryElevateImageFilter_txx

#include "stdio.h"

#include "itkNaryElevateImageFilter.h"

#include "itkMetaDataDictionary.h"

#include "itkImageRegionIteratorWithIndex.h"

namespace itk
{

// helper function to get the value of a specific tag
double gettagvalue(std::string &entryId, MetaDataDictionary &metaDict)
{
  ExceptionObject ex("invalid tag value");

  MetaDataDictionary::ConstIterator tagItr=metaDict.Find(entryId);

  if (tagItr!=metaDict.End())
  {
    typedef MetaDataObject<std::string> MetaDataStringType;
    MetaDataObjectBase::Pointer entry=tagItr->second;
    MetaDataStringType::Pointer entryvalue=
      dynamic_cast<MetaDataStringType *>(entry.GetPointer());

    if (entryvalue)
    {
      std::string tagvalue=entryvalue->GetMetaDataObjectValue();

      float value;
      if (sscanf(tagvalue.c_str(),"%g",&value)!=1) throw ex;
      else return(value);
    }
  }
  else throw ex;

  return(0.0);
}

// constructor
template <class TInputImage, class TOutputImage, class TFunction>
NaryElevateImageFilter<TInputImage, TOutputImage, TFunction>::
NaryElevateImageFilter()
{
  m_DataMin=0.0f;
  m_DataMax=0.0f;
  m_DataMean=0.0f;
  m_NoiseThres=0.0f;
}

// check inputs
template <class TInputImage, class TOutputImage, class TFunction>
void
NaryElevateImageFilter<TInputImage, TOutputImage, TFunction>::
GenerateOutputInformation()
{
  const unsigned int dimI=InputImageDimension;
  const unsigned int dimO=OutputImageDimension;

  if (dimO<dimI+1) itkExceptionMacro(<< "inadequate output image dimension");

  OutputImagePointer outputPtr = this->GetOutput(0);

  const unsigned int numberOfInputImages =
    static_cast<int>(this->GetNumberOfInputs());

  for (unsigned int i=0; i<numberOfInputImages; i++)
  {
    InputImagePointer inputPtr =
      dynamic_cast<InputImageType *>(ProcessObject::GetInput(i));

    InputImageRegionType regionI=inputPtr->GetRequestedRegion();

    InputImageSizeType    sizeI    = regionI.GetSize();
    InputImageIndexType   indexI   = regionI.GetIndex();
    InputImageOriginType  originI  = inputPtr->GetOrigin();
    InputImageSpacingType spacingI = inputPtr->GetSpacing();

    OutputImageSizeType    sizeO;
    OutputImageIndexType   indexO;
    OutputImageOriginType  originO;
    OutputImageSpacingType spacingO;

    if (i==0)
    {
      for (unsigned int j=0; j<dimI; j++)
      {
        sizeO[j]    = sizeI[j];
        indexO[j]   = indexI[j];
        originO[j]  = originI[j];
        spacingO[j] = spacingI[j];
      }

      sizeO[dimI]=numberOfInputImages;
      indexO[dimI]=0;
      originO[dimI]=0;
      spacingO[dimI]=1;

      for (unsigned int j=dimI+1; j<dimO; j++)
      {
        sizeO[j]    = 1;
        indexO[j]   = 0;
        originO[j]  = 0;
        spacingO[j] = 1;
      }

      OutputImageRegionType regionO;
      regionO.SetSize(sizeO);
      regionO.SetIndex(indexO);

      outputPtr->SetOrigin(originO);
      outputPtr->SetSpacing(spacingO);

      outputPtr->SetRegions(regionO);
    }
    else
      for (unsigned int j=0; j<dimI; j++)
        if (sizeO[j]!=sizeI[j] || indexO[j]!=indexI[j])
          itkExceptionMacro(<< "mismatch of nary input images");
  }

  // pass all tags from first time point:

  InputImagePointer inputPtr0 =
    dynamic_cast<InputImageType *>(ProcessObject::GetInput(0));

  outputPtr->SetMetaDataDictionary(inputPtr0->GetMetaDataDictionary());

  // retrieve all time points as list:

  typedef MetaDataDictionary DictionaryType;

  std::list<double> list;

  double firstdate = 0;
  double firsttime = 0;

  int firsthour = 0;
  int firstminutes = 0;
  double firstseconds = 0;

  bool dontusedict=false;

  for (unsigned int i=0; i<numberOfInputImages; i++)
  {
    InputImagePointer inputPtr =
      dynamic_cast<InputImageType *>(ProcessObject::GetInput(i));

    DictionaryType &metaDict=inputPtr->GetMetaDataDictionary();

    if (metaDict.Begin()==metaDict.End() || dontusedict)
    {
      ExceptionObject ex("invalid time series");
      if (i>0 && !dontusedict) throw ex;

      list.push_back(i); // assume that time points are given in minutes
      dontusedict=true;
      continue;
    }

    std::string entryIdAD="0008|0022"; // acquisition date
    std::string entryIdAT="0008|0032"; // acquisition time
    std::string entryIdAN="0020|0012"; // aquisition number

    double entryAD=gettagvalue(entryIdAD,metaDict);
    double entryAT=gettagvalue(entryIdAT,metaDict);

    if (i==0)
    {
      firstdate=entryAD;
      firsttime=entryAT;

      firsthour=(int)(firsttime/10000)%100;
      firstminutes=(int)(firsttime/100)%100;
      firstseconds=firsttime-100*((int)firsttime/100);

      list.push_back(0.0);
    }
    else
    {
      double date=entryAD;
      double time=entryAT;

      int hour=(int)(time/10000)%100;
      int minutes=(int)(time/100)%100;
      double seconds=time-100*((int)time/100);

      // assume that if date differs we have just passed midnight
      if (date!=firstdate)
      {
        hour+=24;
        firstdate=date;
      }

      double difference=(seconds-firstseconds)+
                        60.0*(minutes-firstminutes)+
                        3600.0*(hour-firsthour);

      list.push_back(difference/60.0); // minutes
    }
  }

  // encapsulate list into MetaDataObject and append to MetaDataDictionary:

  typedef MetaDataObject< std::list<double> > MetaDataListType;
  MetaDataListType::Pointer value=MetaDataListType::New();
  value->SetMetaDataObjectValue(list);

  std::string internalId="___internal-4D-filter-time-point-list"; // internal tag
  DictionaryType &metaDict=outputPtr->GetMetaDataDictionary();
  metaDict[internalId]=value;
}

// elevate inputs
template <class TInputImage, class TOutputImage, class TFunction>
void
NaryElevateImageFilter<TInputImage, TOutputImage, TFunction>::
GenerateData()
{
  const unsigned int dimI=InputImageDimension;
  const unsigned int dimO=OutputImageDimension;

  OutputImagePointer outputPtr = this->GetOutput(0);

  const unsigned int numberOfInputImages =
    static_cast<int>(this->GetNumberOfInputs());

  typedef ImageRegionConstIteratorWithIndex<InputImageType> ImageRegionConstIteratorWithIndexType;
  std::vector<ImageRegionConstIteratorWithIndexType *> inputItrVector(numberOfInputImages);

  for (unsigned int i=0; i<numberOfInputImages; i++)
  {
    InputImagePointer inputPtr =
      dynamic_cast<InputImageType *>(ProcessObject::GetInput(i));

    inputPtr->Update();

    ImageRegionConstIteratorWithIndexType *inputItr =
      new ImageRegionConstIteratorWithIndexType(inputPtr, inputPtr->GetBufferedRegion());

    inputItrVector[i] = reinterpret_cast<ImageRegionConstIteratorWithIndexType *>(inputItr);
    inputItrVector[i]->GoToBegin();
  }

  // get time point list from meta dictionary:

  typedef MetaDataDictionary DictionaryType;
  DictionaryType &metaDict=outputPtr->GetMetaDataDictionary();

  std::string internalId="___internal-4D-filter-time-point-list"; // internal tag
  MetaDataDictionary::ConstIterator tagItr=metaDict.Find(internalId);

  ExceptionObject ex("invalid time point list");

  Array<double> timePoints;
  timePoints.SetSize(numberOfInputImages);

  if (tagItr!=metaDict.End())
  {
    typedef MetaDataObject< std::list<double> > MetaDataListType;

    MetaDataObjectBase::Pointer entry = tagItr->second;
    MetaDataListType::Pointer entryvalue = dynamic_cast<MetaDataListType *>( entry.GetPointer() );

    // check whether or not the type of the entry value is correct
    if (entryvalue)
    {
      std::list<double> list;
      list=entryvalue->GetMetaDataObjectValue();

      if (list.size()!=numberOfInputImages) throw ex;

      std::list<double>::const_iterator listItr=list.begin();

      for (unsigned int k=0; k<numberOfInputImages; k++)
      {
        timePoints[k]=*listItr;
        listItr++;
      }
    }
    else throw ex;
  }
  else throw ex;

  // process inputs by retrieving one time series after another:

  outputPtr->Allocate();

  m_DataMin=NumericTraits<float>::max();
  m_DataMax=NumericTraits<float>::min();

  m_DataMean=0.0f;
  unsigned int meancount=0;

  m_NoiseThres=0.0f;
  unsigned int noisecount=0;

  InputImageIndexType indexI;
  OutputImageIndexType indexO;

  for (unsigned int i=dimI+1; i<dimO; i++) indexO[i]=0;

  while(!inputItrVector[0]->IsAtEnd())
  {
    indexI=inputItrVector[0]->GetIndex();

    for (unsigned int j=0; j<dimI; j++) indexO[j]=indexI[j];

    InputImagePixelType base;

    float mean_fvalue=0.0f;

    float mingrad=NumericTraits<float>::max();
    float rndgrad=0.0f;

    float last_fvalue=0.0f,lastlast_fvalue=0.0f;

    NaryInputType values;
    values.set_size(numberOfInputImages);

    // get actual time series
    for (unsigned int inputNumber=0; inputNumber<numberOfInputImages; inputNumber++)
    {
      InputImagePixelType value;
      value=inputItrVector[inputNumber]->Get();

      values[inputNumber]=value;

      ++(*inputItrVector[inputNumber]);
    }

    // push actual time series through elevate functor
    NaryOutputType result;
    result.SetSize(numberOfInputImages);
    m_Functor(values,timePoints,result);

    // check actual time series
    for (unsigned int inputNumber=0; inputNumber<numberOfInputImages; inputNumber++)
    {
      OutputImagePixelType value;
      value=result[inputNumber];

      if (inputNumber==0) base=value;

      float fvalue;
      fvalue=static_cast<float>(value);

      if (fvalue<m_DataMin) m_DataMin=fvalue;
      if (fvalue>m_DataMax) m_DataMax=fvalue;

      mean_fvalue+=fvalue;

      if (inputNumber>1)
      {
        float grad=last_fvalue-lastlast_fvalue;
        if (grad<0.0f) grad=-grad;

        if (grad<mingrad)
        {
          mingrad=grad;
          rndgrad=fvalue-last_fvalue;
          if (rndgrad<0.0f) rndgrad=-rndgrad;
        }
      }

      lastlast_fvalue=last_fvalue;
      last_fvalue=fvalue;
    }

    // store resulting time series
    for (unsigned int inputNumber=0; inputNumber<numberOfInputImages; inputNumber++)
    {
      indexO[dimI]=inputNumber;
      outputPtr->SetPixel(indexO, result[inputNumber]);
    }

    mean_fvalue/=numberOfInputImages;

    if (++meancount>1) m_DataMean*=float(meancount-1)/meancount;
    m_DataMean+=mean_fvalue/meancount;

    m_NoiseThres+=rndgrad;
    noisecount++;
  }

  // clean up:

  for (unsigned int i=0; i<numberOfInputImages; i++) delete inputItrVector[i];

  m_NoiseThres/=noisecount;
}

}

#endif
