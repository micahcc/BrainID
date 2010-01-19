#include "tools.h"

#include <vector>
#include <itkComplexToModulusImageFilter.h>
#include <itkComplexToPhaseImageFilter.h>
#include <itkFFTRealToComplexConjugateImageFilter.h>

void outputVector(std::ostream& out, indii::ml::aux::vector mat) 
{
  unsigned int i;
  for (i = 0; i < mat.size(); i++) {
      out << std::setw(15) << mat(i);
  }
}

void outputMatrix(std::ostream& out, indii::ml::aux::matrix mat) 
{
  unsigned int i, j;
  for (j = 0; j < mat.size2(); j++) {
    for (i = 0; i < mat.size1(); i++) {
      out << std::setw(15) << mat(i,j);
    }
    out << std::endl;
  }
}

void fftline()
{

}

unsigned int round_power_2(unsigned int in) 
{
    unsigned int count = 0;
    while(in != 0) {
        count++;
        in >>= 1;
    }

    //set the size of the temporary var and the buffer
    return(1 << count);
}

void copyTimeLine(itk::OrientedImage<double,4>::Pointer src, 
            itk::Image<double,1>::Pointer dest,
            itk::OrientedImage<double,4>::IndexType pos) 
{
    itk::ImageLinearIteratorWithIndex< itk::Image< double, 1 > >
                it(dest, dest->GetRequestedRegion());
    it.SetDirection(0);
    it.GoToBegin();

    while(!it.IsAtEnd()) {
        if(it.GetIndex()[0] > (int)src->GetRequestedRegion().GetSize()[3])
            break;
        pos[3] = it.GetIndex()[0];
        it.Set(src->GetPixel(pos));
        ++it;
    }
}

void copyTimeLine(itk::Image<double,1>::Pointer src, 
            itk::OrientedImage<double,4>::Pointer dest,
            itk::OrientedImage<double,4>::IndexType pos) 
{
    itk::ImageLinearIteratorWithIndex< itk::Image< double, 1 > >
                it(src, src->GetRequestedRegion());
    it.SetDirection(0);
    it.GoToBegin();

    while(!it.IsAtEnd()) {
        if(it.GetIndex()[0] > (int)dest->GetRequestedRegion().GetSize()[3]) 
            break;
        pos[3] = it.GetIndex()[0];
        dest->SetPixel(pos, it.Get());
        ++it;
    }
}

itk::OrientedImage<double, 4>::Pointer fft_image(
            itk::OrientedImage<double,4>::Pointer inimg)
{
    typedef itk::Image< std::complex<double>, 1> ComplexT;
    typedef itk::OrientedImage< double, 4> Real4DT;
    typedef itk::OrientedImage< double, 1> Real1DTBase;
    typedef itk::Image< double, 1> Real1DT;
    typedef itk::ComplexToModulusImageFilter< ComplexT, Real1DT > ModT;
    typedef itk::ComplexToPhaseImageFilter< ComplexT, Real1DT > PhasT;
    typedef itk::FFTRealToComplexConjugateImageFilter< double, 1 > FFT1DT;
    typedef itk::CastImageFilter< Real1DT, Real1DTBase> castF;

    //Set up sizes and indices to grab a single time vector
    //copy it into another image
    Real4DT::SizeType inSize = inimg->GetRequestedRegion().GetSize();
    Real4DT::SizeType timeSizeIn = {{1, 1, 1, inSize[3]}};
    Real4DT::IndexType index = {{0, 0, 0, 0}};
    Real1DT::SizeType lineSize = {{1}};
    Real4DT::RegionType regionSel;


    //Round up the nearest power of 2 for image length
    //and create a temporary image for FFT's
    lineSize[0] = round_power_2(inSize[3]);
    Real4DT::SizeType timeSizeOut = {{1, 1, 1, lineSize[0]}};
    Real1DT::Pointer working = Real1DT::New();
    working->SetRegions(lineSize);
    working->Allocate();
    castF::Pointer cast = castF::New();
    cast->SetInput(working);
    
    //Setup output image
    Real4DT::Pointer out = Real4DT::New();
    out->SetRegions(timeSizeOut);
    out->Allocate();

    //Create the 1-D filters
    FFT1DT::Pointer fft = FFT1DT::New();
    fft->SetInput(cast->GetOutput());
    
    ModT::Pointer modulus = ModT::New();
    modulus->SetInput(fft->GetOutput());
    PhasT::Pointer phase = PhasT::New();
    phase->SetInput(fft->GetOutput());

    for(index[0] = 0 ; index[0] < (int)inSize[0] ; index[0]++) {
        for(index[1] = 0 ; index[1] < (int)inSize[1] ; index[1]++) {
            for(index[2] = 0 ; index[2] < (int)inSize[2] ; index[2]++) {
                regionSel.SetIndex(index);
                regionSel.SetSize(timeSizeIn);
                copyTimeLine(inimg, working, index);
                
                modulus->Update();
                phase->Update();

                copyTimeLine(modulus->GetOutput(), out, index);
            }
        }
    }

    return out;
}

