#include "tools.h"

#include <vector>
#include <cmath>
#include <itkComplexToModulusImageFilter.h>
#include <itkComplexToPhaseImageFilter.h>
#include <itkFFTRealToComplexConjugateImageFilter.h>

#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkSquareImageFilter.h>

typedef itk::AddImageFilter< Image3DType > AddF3;
typedef itk::SubtractImageFilter< Image3DType > SubF3;
typedef itk::SquareImageFilter< Image3DType, Image3DType > SqrF3;

/* Calculates the percent difference between input1, and input2,
 * using input1 as the reference
 */
Image4DType::Pointer pctDiff(const Image4DType::Pointer input1,
            const Image4DType::Pointer input2)
{
    itk::ImageLinearConstIteratorWithIndex<Image4DType> iter1(
                input1, input1->GetRequestedRegion());
    iter1.SetDirection(3);
    iter1.GoToBegin();
    
    Image4DType::Pointer out = Image4DType::New();
    out->SetRegions(input1->GetRequestedRegion());
    out->Allocate();
    out->FillBuffer(1);
    
    itk::ImageLinearIteratorWithIndex<Image4DType> itero(
                out, out->GetRequestedRegion());
    itero.SetDirection(3);
    itero.GoToBegin();

    Image4DType::SpacingType space = input2->GetSpacing();
    space[3] = input1->GetSpacing()[3];
    input2->SetSpacing(space);

    while(!iter1.IsAtEnd()) {
        while(!iter1.IsAtEndOfLine()) {
            Image4DType::PointType point;
            input1->TransformIndexToPhysicalPoint(iter1.GetIndex(), point);
            Image4DType::IndexType index;
            input2->TransformPhysicalPointToIndex(point, index);
            if(input2->GetRequestedRegion().IsInside(index)) {
                printf("%li %li %li %li\n", iter1.GetIndex()[0], iter1.GetIndex()[1],
                            iter1.GetIndex()[2], iter1.GetIndex()[3]);
                itero.Set((input2->GetPixel(index)-iter1.Get())/input2->GetPixel(index));
            }
            ++iter1; ++itero;
        }
        iter1.NextLine();
        itero.NextLine();
    }
    return out;
}

Image3DType::Pointer mse(const Image4DType::Pointer input1,
            const Image4DType::Pointer input2)
{
    if(input1->GetRequestedRegion().GetSize() != 
                input2->GetRequestedRegion().GetSize()) {
        return NULL;
    }
    
    itk::ImageLinearConstIteratorWithIndex<Image4DType> iter1(
                input1, input1->GetRequestedRegion());
    iter1.SetDirection(3);
    iter1.GoToBegin();
    
    itk::ImageLinearConstIteratorWithIndex<Image4DType> iter2(
                input2, input2->GetRequestedRegion());
    iter2.SetDirection(3);
    iter2.GoToBegin();

    Image4DType::IndexType index4;
    Image3DType::Pointer out = Image3DType::New();
    Image4DType::SizeType size4 = input1->GetRequestedRegion().GetSize();
    {
        Image3DType::SizeType size3 = {{size4[0], size4[1], size4[2]}};
        out->SetRegions(size3);
        out->Allocate();
    }

    while(!iter1.IsAtEnd() && !iter2.IsAtEnd()) {
        index4 = iter1.GetIndex();
        double mean = 0;
        while(!iter1.IsAtEndOfLine() && !iter2.IsAtEndOfLine()) {
            mean += pow(iter1.Get() - iter2.Get(), 2);
            ++iter1; ++iter2;
        }
        index4 = iter1.GetIndex();
        Image3DType::IndexType index3 = {{index4[0], index4[1], index4[2]}};
        out->SetPixel(index3, mean/size4[3]);
        iter1.NextLine();
        iter2.NextLine();
    }
    return out;
}

std::vector<Activation> read_activations(const char* filename)
{
    FILE* fin = fopen(filename, "r");
    std::vector< Activation > output;
    if(!fin) {
        fprintf(stderr, "read_activations: \"%s\" is invalid\n", filename);
        return output;
    }
    
    char* input = NULL;
    size_t size = 0;
    char* curr = NULL;
    double prev = 1/0.;
    Activation parsed;
    printf("Parsing activations\n");
    while(getline(&input, &size, fin) && !feof(fin)) {
        parsed.time = strtod(input, &curr);
        parsed.level = strtod(curr, NULL);
        
        if(!(prev == parsed.level)) {
            output.push_back(parsed);
        }
        prev = parsed.level;
        free(input);
        input = NULL;
    }
    fclose(fin);
    return output;
}


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
    lineSize[0] = round_power_2(inSize[3])*4;
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
};

Image3DType::Pointer Tvar(const Image4DType::Pointer fmri_img)
{
    Image3DType::Pointer mean = Tmean(fmri_img);
    itk::ImageLinearConstIteratorWithIndex<Image4DType> iter(
                fmri_img, fmri_img->GetRequestedRegion());
    iter.SetDirection(3);
    iter.GoToBegin();
    Image4DType::IndexType index4;
    Image3DType::Pointer out = Image3DType::New();
    Image4DType::SizeType size4 = fmri_img->GetRequestedRegion().GetSize();
    {
        Image3DType::SizeType size3 = {{size4[0], size4[1], size4[2]}};
        out->SetRegions(size3);
        out->Allocate();
    }

    while(!iter.IsAtEnd()) {
        index4 = iter.GetIndex();
        Image3DType::IndexType index3 = {{index4[0], index4[1], index4[2]}};
        double average = 0;
        while(!iter.IsAtEndOfLine()) {
            average += pow(iter.Get()-mean->GetPixel(index3),2);
            ++iter;
        }
        out->SetPixel(index3, average/size4[3]);
        iter.NextLine();
    }
    return out;
}

Image3DType::Pointer Tmean(const Image4DType::Pointer fmri_img)
{
    itk::ImageLinearConstIteratorWithIndex<Image4DType> iter(
                fmri_img, fmri_img->GetRequestedRegion());
    iter.SetDirection(3);
    iter.GoToBegin();
    Image4DType::IndexType index4;
    Image3DType::Pointer out = Image3DType::New();
    Image4DType::SizeType size4 = fmri_img->GetRequestedRegion().GetSize();
    {
        Image3DType::SizeType size3 = {{size4[0], size4[1], size4[2]}};
        out->SetRegions(size3);
        out->Allocate();
    }

    while(!iter.IsAtEnd()) {
        index4 = iter.GetIndex();
        double average = 0;
        while(!iter.IsAtEndOfLine()) {
            average += iter.Get();
            ++iter;
        }
        index4 = iter.GetIndex();
        Image3DType::IndexType index3 = {{index4[0], index4[1], index4[2]}};
        out->SetPixel(index3, average/size4[3]);
        iter.NextLine();
    }
    return out;
}

//Image3DType::Pointer Tvar(const Image4DType::Pointer fmri_img)
//{
//    Image3DType::Pointer average = Tmean(fmri_img);
//
//    /* Used to zero out the addfilter */
//    Image3DType::Pointer zero = Image3DType::New();
//    zero->SetRegions(average->GetRequestedRegion());
//    zero->Allocate();
//    zero->FillBuffer(0);
//    
//    /* Initialize the Addition */
//    AddF3::Pointer add = AddF3::New();
//    add->GraftOutput(zero);
//    add->SetInput2(add->GetOutput());
//    
//    /* Initialize Subtraction */
//    SubF3::Pointer sub = SubF3::New();   
//    
//    /* Initialize Subtraction */
//    SqrF3::Pointer sqr = SqrF3::New();   
//    
//    /* Calculate Sum of Images */
//    //SUM( (X_i - mu)^2 )
//    for(size_t ii = 0 ; ii < fmri_img->GetRequestedRegion().GetSize()[3] ; ii++) {
//        sub->SetInput1(extract(fmri_img, ii));
//        sub->SetInput2(average);
//        sqr->SetInput(sub->GetOutput());
//        add->SetInput1(sqr->GetOutput());
//        add->Update();
//    }
//
//    /* Calculate Average of Images */
//    ScaleF::Pointer scale = ScaleF::New();
//    scale->SetInput(add->GetOutput());
//    scale->SetConstant(1./fmri_img->GetRequestedRegion().GetSize()[3]);
//    scale->Update();
//    return scale->GetOutput();
//}


Image4DType::Pointer extrude(const Image3DType::Pointer input, unsigned int len)
{
    itk::ImageLinearConstIteratorWithIndex<Image3DType> iter(
                input, input->GetRequestedRegion());
    iter.SetDirection(0);
    iter.GoToBegin();
    Image3DType::SizeType size3 = input->GetRequestedRegion().GetSize();

    Image4DType::Pointer out = Image4DType::New();
    {
        Image4DType::SizeType size4 = {{size3[0], size3[1], size3[2], len}};
        out->SetRegions(size4);
        out->Allocate();
    }

    while(!iter.IsAtEnd()) {
        while(!iter.IsAtEndOfLine()) {
            Image3DType::IndexType index3 = iter.GetIndex();
            for(unsigned int i = 0 ; i < len ; i++) {
                Image4DType::IndexType index4 = {{index3[0], index3[1], index3[2], i}};
                out->SetPixel(index4, iter.Get());
            }
            ++iter;
        }
        iter.NextLine();
    }
    return out;
}

//RMS for a non-zero mean signal is 
//sqrt(mu^2+sigma^2)
Image3DType::Pointer get_rms(Image4DType::Pointer in)
{ 
    Image3DType::Pointer out = Image3DType::New();
    Image3DType::SizeType outsize;
    Image3DType::DirectionType outdir;
    Image3DType::PointType outorigin;
    Image3DType::SpacingType outspacing;
    for(int i = 0 ; i < 3 ; i++) {
        outsize[i] = in->GetRequestedRegion().GetSize()[i];
        outorigin[i] = in->GetOrigin()[i];
        outspacing[i] = in->GetSpacing()[i];
        
        for(int j = 0 ; j < 3 ; j++) 
            outdir(i, j) = in->GetDirection()(i,j);
    }
    out->SetRegions(outsize);
    out->SetDirection(outdir);
    out->SetOrigin(outorigin);
    out->SetSpacing(outspacing);
    out->Allocate();
    
    for(size_t xx = 0 ; xx < in->GetRequestedRegion().GetSize()[0] ; xx++) {
        for(size_t yy = 0 ; yy < in->GetRequestedRegion().GetSize()[1] ; yy++) {
            for(size_t zz = 0 ; zz < in->GetRequestedRegion().GetSize()[2] ; zz++) {
                double mean = 0;
                double var = 0;
                
                //calc mean
                for(size_t tt = 0 ; tt < in->GetRequestedRegion().GetSize()[3] ; tt++) {
                    Image4DType::IndexType index = {{xx, yy, zz, tt}};
                    mean += in->GetPixel(index);
                }
                mean /= in->GetRequestedRegion().GetSize()[3];
                
                //calc cov
                for(size_t tt = 0 ; tt < in->GetRequestedRegion().GetSize()[3] ; tt++) {
                    Image4DType::IndexType index = {{xx, yy, zz, tt}};
                    var += pow(in->GetPixel(index)-mean, 2);
                }
                var /= in->GetRequestedRegion().GetSize()[3];
                
                //set output pixel as rms
                {
                    Image3DType::IndexType index = {{xx, yy, zz}};
                    out->SetPixel(index, sqrt(mean*mean + var));
                }
            }
        }
    }

    return out;
}
