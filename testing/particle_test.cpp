#include <itkVector.h>
#include "smctc.hh"

#include <cstdio> 
#include <cstdlib>
#include <cstring>

using namespace std;

int main(int argc, char** argv)
{
    
    if(argc != 3) {
        printf("Usage: %s <inputname> <outputname>", argv[0])
    }
    
    long lNumber = 1000;
    long lIterates;

    //Load observations
    typedef    float ImagePixelType;
    typedef itk::Image< ImagePixelType,  2 > ImageType;
    typedef itk::ImageFileReader< ImageType >  ImageReaderType;
    typedef itk::ImageFileWriter< ImageType >  WriterType;
    
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    //Initialise and run the sampler
    smc::sampler<cv_state> Sampler(lNumber, SMC_HISTORY_NONE);  
    smc::moveset<cv_state> Moveset(fInitialise, fMove, NULL);

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

long load_data(char const * szName, cv_obs** yp)
{
  FILE * fObs = fopen(szName,"rt");
  if (!fObs)
    throw SMC_EXCEPTION(SMCX_FILE_NOT_FOUND, "Error: pf assumes that the current directory contains an appropriate data file called data.csv\nThe first line should contain a constant indicating the number of data lines it contains.\nThe remaining lines should contain comma-separated pairs of x,y observations.");
  char* szBuffer = new char[1024];
  fgets(szBuffer, 1024, fObs);
  long lIterates = strtol(szBuffer, NULL, 10);

  *yp = new cv_obs[lIterates];
  
  for(long i = 0; i < lIterates; ++i)
    {
      fgets(szBuffer, 1024, fObs);
      (*yp)[i].x_pos = strtod(strtok(szBuffer, ",\r\n "), NULL);
      (*yp)[i].y_pos = strtod(strtok(NULL, ",\r\n "), NULL);
    }
  fclose(fObs);

  delete [] szBuffer;

  return lIterates;
}

double integrand_mean_x(const cv_state& s, void *)
{
  return s.x_pos;
}

double integrand_var_x(const cv_state& s, void* vmx)
{
  double* dmx = (double*)vmx;
  double d = (s.x_pos - (*dmx));
  return d*d;
}

double integrand_mean_y(const cv_state& s, void *)
{
  return s.y_pos;
}

double integrand_var_y(const cv_state& s, void* vmy)
{
  double* dmy = (double*)vmy;
  double d = (s.y_pos - (*dmy));
  return d*d;
}
