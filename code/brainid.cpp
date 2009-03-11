#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>

#include <indii/ml/filter/ParticleFilter.hpp>
//#include "ParticleFilter.hpp"
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
//#include "StratifiedParticleResampler.hpp"
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

//class ParticleFilterMod : public ParticleFilter<double>
//{
//public:
//    void print_particles() {
//        for (unsigned int i = 0; i < this->p_xtn_ytn.getSize(); i++) {
//            outputVector(std::cerr, this->p_xtn_ytn.get(i));
//        }
//    }
//
//    ParticleFilterMod(ParticleFilterModel<double>* model,
//            indii::ml::aux::DiracMixturePdf& p_x0);
//
//    virtual void filter(const double tnp1, const indii::ml::aux::vector& ytnp1);
//};
//
//void ParticleFilterMod::filter(const double tnp1, const aux::vector& ytnp1) {
//    
//};
//
//ParticleFilterMod::ParticleFilterMod(ParticleFilterModel<double>* model,
//            indii::ml::aux::DiracMixturePdf& p_x0) : 
//            ParticleFilter<double>(model, p_x0) {
//
//};


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
  aux::GaussianPdf suggestPrior(const aux::vector& init);

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
//  var_e = 3.92e-6;
  var_e = 3.92e-2;
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
    double v_t = s(0);
    static aux::symmetric_matrix cov(1);
    cov(0,0) = .001;
    static aux::GaussianPdf rng(aux::zero_vector(1), cov);
    
    //This is a bit of a kludge, but it is unavoidable right now
    if(s(0) < 0) {
        fprintf(stderr, "Warning, had to move volume to");
        fprintf(stderr, "zero because it was negative\n");
        v_t = 0;
    }


    //TODO, potentially add some randomness here.
    //std::cerr << "Printing input state" << endl;
    //outputVector(std::cerr, s);
    //std::cerr << endl;

    aux::vector w(SYSTEM_SIZE);
    //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
    double dot = (1./tau_0) * (s(3) - pow(v_t, 1./alpha)) + (rng.sample())[0];
    w(0) = v_t + dot*delta;
    if(w(0) < 0) w(0) = 0;
    
    //Q_t* = ...
    double A = (s(3) / E_0) * (1 - pow( 1. - E_0, 1./s(3)));
    double B = s(1) / pow(v_t, 1.-1./alpha);
    dot = (1./tau_0) * ( A - B );
    w(1) = s(1) + dot*delta;

    //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
    dot = u(0)*epsilon - s(2)/tau_s - (s(3) - 1.) / tau_f;
    w(2) = s(2) + dot*delta;

    //f_t* = s_t;
    dot = s(2);
    w(3) = s(3) + dot*delta;

    //std::cerr  <<"Printing output state" << endl;
    outputVector(std::cerr, w);
    std::cerr << endl;
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
    cov(0,0) = 1;
    static aux::GaussianPdf rng(aux::zero_vector(1), cov);
    
    aux::vector location(1);
    location(0) = 0;
    location(0) = y(0);
    fprintf(stderr, "Actual: %f\n", location(0));
    fprintf(stderr, "Measure: %f\n", (measure(s))(0));
    fprintf(stderr, "Particle:\n");
    outputVector(std::cerr , s);
    fprintf(stderr, ":\n");
    location(0) -= (measure(s))(0);
    location(0) /= var_e;
    fprintf(stderr, "Location calculated: %f\n", location(0));
    double out = rng.densityAt(location);
    fprintf(stderr, "Weight calculated: %e\n", out);
    return out;
}

aux::GaussianPdf BoldModel::suggestPrior()
{
    aux::vector mu(SYSTEM_SIZE);
    mu.clear();
    return suggestPrior(mu);
}

aux::GaussianPdf BoldModel::suggestPrior(const aux::vector& init)
{
    aux::symmetric_matrix sigma(SYSTEM_SIZE);

    sigma.clear();
    sigma(0,0) = 10.0;
    sigma(1,1) = 10.0;
    sigma(2,2) = 10.0;
    sigma(3,3) = 10.0;

    return aux::GaussianPdf(init, sigma);
}

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
//    aux::vector location(1);
//    aux::symmetric_matrix cov(1);
//    cov(0,0) = 1;
//    aux::GaussianPdf rng(aux::zero_vector(1), cov);
//    for(int i=0 ; i<10 ; i++) {
//        location(0) = i/10.;
//        fprintf(stderr, "Density at %f: %f\n", location(0), rng.densityAt(location));
//    }
    
    if(argc != 2) {
        printf("Usage: %s <inputname>\n", argv[0]);
        return -1;
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
    index[0] = 1; //skip section label
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
    fpred << "# columns: 11" << endl;
    
    fpred << t << ' ';
    outputVector(fpred, mu);
    fpred << endl;

    aux::vector sample_state(SYSTEM_SIZE);

//TODO, get resample working
//TODO, get distribution creation working
//TODO, stop using ABS for s(0)
    while(!iter.IsAtEndOfLine()) {
        fprintf(stderr, "Size0: %u\n", pred.getSize());
        meas(0) = iter.Get();
        ++iter;
    
        std::cerr << "Time " << t << endl;
        filter.filter(t, meas);

        fprintf(stderr, "Size1: %u\n", pred.getSize());
        filter.resample(&resampler);
        pred = filter.getFilteredState();
        fprintf(stderr, "Size2: %u\n", pred.getSize());
        mu = pred.getDistributedExpectation();
//        sample_state = pred.sample();

        /* output measurement */
        fmeas << t << ' ';
        outputVector(fmeas, meas);
        fmeas << endl;

        /* output filtered state */
        fpred << t << ' ';
        outputVector(fpred, mu);
//        fpred << ' ';
//        outputVector(fpred, sample_state);
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

