#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
#include <indii/ml/aux/DiracMixturePdf.hpp>
#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#define SYSTEM_SIZE 2
#define MEAS_SIZE 1
#define ACTUAL_SIZE 3
#define STEPS 250
#define NUM_PARTICLES 1000

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

//States:
//0 - v_t
//1 - q_t
//2 - s_t
//3 - f_t

class BoldModel : ParticleFilterModel<double>
{
public:
  ~BoldModel();
  BoldModel();
  
  unsigned int getStateSize() { return 4; };
  unsigned int getMeasurementSize() { return MEAS_SIZE; };

  aux::vector transition(const aux::vector& s,
      const double t, const double delta);

  aux::vector measure(const aux::vector& s);

  double weight(const aux::vector& s,
      const aux::vector& y);

private:
  double V_0;
  double a_1;
  double a_2;
};

BoldModel::BoldModel()
{
    
}

BoldModel::~BoldModel()
{

}

aux::vector BoldModel::transition(const aux::vector& s,
        const double t, const double delta)
{
    aux::vector w(SYSTEM_SIZE);
    //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
    s(0) = (1./params->named.tau_0) * (state->named.f_t -
                pow(state->named.v_t, 1./params->named.alpha));
    
    //Q_t* = ...
    double A = (state->named.f_t / params->named.e_0) * 
                (1 - pow( 1. - params->named.e_0, 1./state->named.f_t));
    double B = state->named.q_t / pow(state->named.v_t, 1.-1./params->named.alpha);
    tmp.named.q_t = (1./params->named.tau_0) * ( A - B );

    //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
    tmp.named.s_t = params->named.epsilon*extradata->control - 
                state->named.s_t/params->named.tau_s - 
                (state->named.f_t - 1.) / params->named.tau_f;

    tmp.named.f_t = state->named.s_t;

    //X_t = X_{t-1} + dx*dt
    for(int i = 0 ; i < 4 ; i++) {
        dist[i][0] = state->array[i] + tmp.array[i]*extradata->delta_t;
        dist[i][1] = extradata->gg*extradata->delta_t;
    }
    
}

aux::vector BoldModel::measure(const aux::vector& s)
{
    aux::vector y(MEAS_SIZE);
    y(0) = V_0 * ( a_1 * ( 1 - s(1)) - a_2 * (1 - s(0)));
    return y;
}

double BoldModel::weight(const aux::vector& s, const aux::vector& y)
{

}

int main(int argc, char* argv[])
{
    if(argc != 3) {
        printf("Usage: %s <inputname> <outputname>", argv[0]);
    }
    
    /* Open up the input */
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    /* Create a model */
    BoldModel model(); 

    /* Create the filter */

}
