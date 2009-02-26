#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkVector.h>
#include "bold.h"

#include <cstdio> 
#include <cstdlib>
#include <cstring>

using namespace std;

//Load observations
typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

int main(int argc, char** argv)
{
    
    if(argc != 3) {
        printf("Usage: %s <inputname> <outputname>", argv[0]);
    }
    
    long lNumber = 1000;
    long lIterates;

    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    //Initialise and run the sampler
    smc::sampler<State_t> Sampler(lNumber, SMC_HISTORY_NONE);
    smc::moveset<State_t> Moveset(fInitialize, fMove, NULL);

    Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL, 0.5);
    Sampler.SetMoveSet(Moveset);
    Sampler.Initialise();
    
    for(int n=1 ; n < lIterates ; ++n) {
      Sampler.Iterate();
      
      double xm,xv,ym,yv;
      xm = Sampler.Integrate(integrand_mean_x,NULL);
      xv = Sampler.Integrate(integrand_var_x, (void*)&xm);
      ym = Sampler.Integrate(integrand_mean_y,NULL);
      yv = Sampler.Integrate(integrand_var_y, (void*)&ym);
      
      cout << xm << "," << ym << "," << xv << "," << yv << endl;

  catch(smc::exception  e)
    {
      cerr << e;
      exit(e.lCode);
    }
}
