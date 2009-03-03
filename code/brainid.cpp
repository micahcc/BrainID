#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>

#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <iostream>

#define SYSTEM_SIZE 4
#define MEAS_SIZE 1
#define INPUT_SIZE 1
#define ACTUAL_SIZE 1
#define STEPS 250
#define NUM_PARTICLES 1000

#define A1 3.4
#define A2 1.0

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

void outputVector(ostream& out, aux::vector vec);

//States:
//0 - v_t
//1 - q_t
//2 - s_t
//3 - f_t

class BoldModel : public ParticleFilterModel<double>
{
public:
  ~BoldModel();
  BoldModel();
  
  unsigned int getStateSize() { return 4; };
  unsigned int getStimSize() { return 1; };
  unsigned int getMeasurementSize() { return MEAS_SIZE; };

  aux::vector transition(const aux::vector& s,
      const double t, const double delta);
  aux::vector transition(const aux::vector& s,
      const double t, const double delta, const aux::vector& u);

  aux::vector measure(const aux::vector& s);

  double weight(const aux::vector& s, const aux::vector& y);

  aux::GaussianPdf suggestPrior();

private:
  double V_0;
  double a_1;
  double a_2;
  double tau_0;
  double tau_s;
  double tau_f;
  double alpha;
  double E_0;
  double epsilon;
  double var_e;
  double small_g;
};

BoldModel::BoldModel()
{
  tau_s = 4.98;
  tau_f = 8.31;
  epsilon = 0.069;
  tau_0 = 8.38;
  alpha = .189;
  E_0 = .635;
  V_0 = 1.49e-2;
  small_g = .95e-5;
  var_e = 3.92e-6;
}

BoldModel::~BoldModel()
{

}

aux::vector BoldModel::transition(const aux::vector& s,
        const double t, const double delta)
{
    aux::zero_vector u(INPUT_SIZE);
    return transition(s, t, delta, u);
}

aux::vector BoldModel::transition(const aux::vector& s,
        const double t, const double delta, const aux::vector& u)
{
    //TODO, potentially add some randomness here.
    //std::cerr << "Printing input state" << endl;
    //outputVector(std::cerr, s);
    //std::cerr << endl;

    aux::vector w(SYSTEM_SIZE);
    //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
    //TODO: THIS IS WRONG SHOULD NOT JUST TO ABS OF S(0)
    double dot = (1./tau_0) * (s(3) - pow(abs(s(0)), 1./alpha));
    w(0) = s(0) + dot*delta;
    
    //Q_t* = ...
    double A = (s(3) / E_0) * (1 - pow( 1. - E_0, 1./s(3)));
    //TODO: THIS IS WRONG SHOULD NOT JUST TO ABS OF S(0)
    double B = s(1) / pow(abs(s(0)), 1.-1./alpha);
    dot = (1./tau_0) * ( A - B );
    w(1) = s(1) + dot*delta;

    //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
    dot = u(0)*epsilon - s(2)/tau_s - (s(3) - 1.) / tau_f;
    w(2) = s(2) + dot*delta;

    //f_t* = s_t;
    dot = s(2);
    w(3) = s(3) + dot*delta;

    //std::cerr  <<"Printing output state" << endl;
    //outputVector(std::cerr, w);
    //std::cerr << endl;
    return w;
}

aux::vector BoldModel::measure(const aux::vector& s)
{
    aux::vector y(MEAS_SIZE);
    y(0) = V_0 * ( A1 * ( 1 - s(1)) - A2 * (1 - s(0)));
    return y;
}

double BoldModel::weight(const aux::vector& s, const aux::vector& y)
{
    //these are really constant throughout the execution
    //of the program, so no need to calculate over and over
    static aux::symmetric_matrix cov(1);
    cov(0,0) = var_e;
    static aux::GaussianPdf rng(aux::zero_vector(1), cov);
    
    aux::vector location(1);
    location(0) = y(0) - measure(s)(0);
    
    return rng.densityAt(location);
}

aux::GaussianPdf BoldModel::suggestPrior()
{
    aux::vector mu(SYSTEM_SIZE);
    aux::symmetric_matrix sigma(SYSTEM_SIZE);

    mu.clear();
    sigma.clear();
    sigma(0,0) = 1.0;
    sigma(1,1) = 1.0;
    sigma(2,2) = 1.0;
    sigma(3,3) = 1.0;

    return aux::GaussianPdf(mu, sigma);
}

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    if(argc != 2) {
        printf("Usage: %s <inputname>", argv[0]);
    }
    
    /* Open up the input */
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    /* Create the iterator, to move forward in time for a particlular section */
    itk::ImageLinearIteratorWithIndex<ImageType> 
        iter(reader->GetOutput(), reader->GetOutput()->GetRequestedRegion());
    iter.SetDirection(1);
    ImageType::IndexType index;
    index[0] = 0;
    index[1] = 5;
    iter.SetIndex(index);

    /* Create a model */
    BoldModel model; 
    aux::GaussianPdf prior = model.suggestPrior();
    aux::DiracMixturePdf x0(prior, NUM_PARTICLES);

    /* Create the filter */
    ParticleFilter<double> filter(&model, x0);
  
    /* create resamplers */
    StratifiedParticleResampler resampler(NUM_PARTICLES);
  
    /* estimate and output results */
    aux::vector meas(MEAS_SIZE);
    aux::DiracMixturePdf pred(SYSTEM_SIZE);
    aux::vector mu(SYSTEM_SIZE);

    pred = filter.getFilteredState();
    mu = pred.getDistributedExpectation();
  
    ofstream fmeas("ParticleFilterHarness_meas.out");
    ofstream fpred("ParticleFilterHarness_filter.out");
    
    double t = .5;
    
    fmeas << "# Created by brainid" << endl;
    fmeas << "# name: measured" << endl;
    fmeas << "# type: matrix" << endl;
    fmeas << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] << endl;
    fmeas << "# columns: 2" << endl;

    fpred << "# Created by brainid" << endl;
    fpred << "# name: measured" << endl;
    fpred << "# type: matrix" << endl;
    fpred << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] << endl;
    fpred << "# columns: 5" << endl;
    
    fpred << t << ' ';
    outputVector(fpred, mu);
    fpred << endl;

    aux::vector sample_state(SYSTEM_SIZE);

//TODO, get resample working
//TODO, get distribution creation working
//TODO, stop using ABS for s(0)
    while(!iter.IsAtEndOfLine()) {
        meas(0) = iter.Get();
        ++iter;
    
        std::cerr << "Time " << t << endl;
        filter.filter(t, meas);
//        filter.resample(&resampler);
        pred = filter.getFilteredState();
//        mu = pred.getDistributedExpectation();
        sample_state = pred.sample();

        /* output measurement */
        fmeas << t << ' ';
        outputVector(fmeas, meas);
        fmeas << endl;

        /* output filtered state */
        fpred << t << ' ';
//        outputVector(fpred, mu);
        outputVector(fpred, sample_state);
        fpred << endl;
        t += .5;
    }

    fmeas.close();
    fpred.close();

  return 0;

}

void outputVector(ostream& out, aux::vector vec) {
  aux::vector::iterator iter, end;
  iter = vec.begin();
  end = vec.end();
  while (iter != end) {
    out << *iter;
    iter++;
    if (iter != end) {
      out << ' ';
    }
  }
}

