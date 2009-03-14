//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan

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

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;
    
const int NUM_PARTICLES = 1000;

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


//State Consists of Two or More Sections:
//Theta
//0 - V_0
//1 - a_1
//2 - a_2
//3 - tau_0
//4 - tau_s
//5 - tau_f
//6 - alpha
//7 - E_0
//8 - epsilon
//Actual States
//9+4*i+0 - v_t
//9+4*i+1 - q_t
//9+4*i+2 - s_t
//9+4*i+3 - f_t

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
    
    //Constants
    static const int THETA_SIZE = 7;
    static const int STATE_SIZE = 4;
    static const int SIMUL_STATES = 1;
    static const int SYSTEM_SIZE = 11;

    static const int MEAS_SIZE = 1;
    static const int INPUT_SIZE = 1;
    static const int STEPS = 250;
    
private:
    aux::vector theta_sigmas;

    inline int indexof(int name, int index){
        return THETA_SIZE + index*STATE_SIZE + name;
    };

    //Internal Constants
    static const double A1 = 3.4;
    static const double A2 = 1.0;
    enum Theta { TAU_S, TAU_F, EPSILON, TAU_0, ALPHA, E_0, V_0};
    enum StateVar { V_T, Q_T, S_T, F_T };
    double var_e;
    double sigma_e;
    double small_g;
};

BoldModel::BoldModel() : theta_sigmas(THETA_SIZE)
{
    if(THETA_SIZE + STATE_SIZE*SIMUL_STATES != SYSTEM_SIZE) {
        std::cerr << "Incorrect system size" << std::endl;
        exit(-1);
    }

    theta_sigmas(TAU_S) = 1.07/4;
    theta_sigmas(TAU_F) = 1.51/4;
    theta_sigmas(EPSILON) = .014/4;
    theta_sigmas(TAU_0) = 1.5/4;
    theta_sigmas(ALPHA) = .004/4;
    theta_sigmas(E_0) = .072/4;
    theta_sigmas(V_0) = .006/4;

    small_g = .95e-5;
    //  var_e = 3.92e-6;
    var_e = 3.92e-3;
    sigma_e = sqrt(var_e);
}

BoldModel::~BoldModel()
{

}

//TODO, I would like to modify these functions so that the vector s
//will just be modified in place, which would reduce the amount of copying
//necessary
aux::vector BoldModel::transition(const aux::vector& s,
        const double t, const double delta)
{
    aux::zero_vector u(INPUT_SIZE);
    return transition(s, t, delta, u);
}

aux::vector BoldModel::transition(const aux::vector& dustin,
        const double time, const double delta_t, const aux::vector& u_t)
{
    aux::vector dustout(SYSTEM_SIZE);
    static aux::symmetric_matrix cov(1);
    cov(0,0) = .001;
    static aux::GaussianPdf rng(aux::zero_vector(1), cov);
    double v_t;
 
    //transition the parameters
    //the GaussianPdf class is a little shady for non univariate, zero-mean
    //cases, so we will just sample from the N(0,1) case and then convert
    //the variables to a correct variance by multiplying by the std-dev
    //The std-deviations are 1/2 the stated std-deviations listed in 
    //Johnston et al.
    dustout(TAU_S)   = dustin(TAU_S)   + rng.sample()[0] * theta_sigmas(TAU_S);
    dustout(TAU_F)   = dustin(TAU_F)   + rng.sample()[0] * theta_sigmas(TAU_F);
    dustout(EPSILON) = dustin(EPSILON) + rng.sample()[0] * theta_sigmas(EPSILON);
    dustout(TAU_0)   = dustin(TAU_0)   + rng.sample()[0] * theta_sigmas(TAU_0);
    dustout(ALPHA)   = dustin(ALPHA)   + rng.sample()[0] * theta_sigmas(ALPHA);
    dustout(E_0)     = dustin(E_0)     + rng.sample()[0] * theta_sigmas(E_0);
    dustout(V_0)     = dustin(V_0)     + rng.sample()[0] * theta_sigmas(V_0);

    //transition the actual state variables
    //TODO, potentially add some randomness here.
    for(int ii=0 ; ii<SYSTEM_SIZE ; ii++) {
        //This is a bit of a kludge, but it is unavoidable right now
        //once this function returns void and dustin isn't const this
        //won't be as necessary, or this could be done when the prior
        //is generated
        v_t = dustin[indexof(V_T,ii)];
        if(v_t < 0) {
            fprintf(stderr, "Warning, had to move volume to");
            fprintf(stderr, "zero because it was negative\n");
            v_t = 0;
        }
        //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
        double dot = (dustin[indexof(F_T,ii)] - pow(v_t, 1./dustin[ALPHA])) / 
                    dustin[TAU_0] + (rng.sample())[0];
        dustout[indexof(V_T,ii)] = v_t + dot*delta_t;
        if(dustout[indexof(V_T,ii)] < 0)
            dustout[indexof(V_T,ii)] = 0;

        //Q_t* = \frac{1}{tau_0} * (\frac{f_t}{E_0} * (1- (1-E_0)^{1/f_t}) - 
        //              \frac{q_t}{v_t^{1-1/\alpha})
        double tmpA = (dustin[indexof(F_T,ii)] / dustin[E_0]) * 
                    (1 - pow( 1. - dustin[E_0], 1./dustin[indexof(F_T,ii)]));
        double tmpB = dustin[indexof(Q_T,ii)] / pow(v_t, 1.-1./dustin[ALPHA]);
        dot =  ( tmpA - tmpB )/dustin[TAU_0];
        dustout[indexof(Q_T,ii)] = dustin[indexof(Q_T,ii)] + dot*delta_t;

        //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
        dot = u_t[0]*dustin[EPSILON]- dustin[indexof(S_T,ii)]/dustin[TAU_S] - 
                    (dustin[indexof(F_T,ii)] - 1.) / dustin[TAU_F];
        dustout[indexof(S_T,ii)] = dustin[indexof(S_T,ii)] + dot*delta_t;

        //f_t* = s_t;
        dot = dustin[indexof(S_T,ii)];
        dustout[indexof(F_T,ii)] = dustin[indexof(F_T,ii)] + dot*delta_t;
    }

    //std::cerr  <<"Printing output state" << endl;
//    outputVector(std::cerr, w);
//    std::cerr << endl;
    return dustout;
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
    location(0) /= sigma_e;
    fprintf(stderr, "Location calculated: %f\n", location(0));
    double out = rng.densityAt(location);
    fprintf(stderr, "Weight calculated: %e\n", out);
    return out;
}

//TODO make some of these non-gaussian
aux::GaussianPdf BoldModel::suggestPrior()
{
    //set the averages of the variables
    aux::vector mu(SYSTEM_SIZE);
    mu[TAU_S] = 4.98;
    mu[TAU_F] = 8.31;
    mu[EPSILON] = 0.069;
    mu[TAU_0] = 8.38;
    mu[ALPHA] = .189;
    mu[E_0] = .635;
    mu[V_0] = 1.49e-2;
    for(int ii = THETA_SIZE ; ii < STATE_SIZE; ii++) {
        mu[ii] = 1;
    }

    //set the variances, assume independence between the variables
    aux::symmetric_matrix sigma(SYSTEM_SIZE);
    sigma.clear();
    sigma(TAU_S, TAU_S) = 1.07;
    sigma(TAU_F, TAU_F) = 1.51;
    sigma(EPSILON, EPSILON) = .014;
    sigma(TAU_0, TAU_0) = 1.50;
    sigma(ALPHA, ALPHA) = .004;
    sigma(E_0, E_0) = .072;
    sigma(V_0, V_0) = .006;

    for(int ii = THETA_SIZE ; ii < STATE_SIZE; ii++) {
        sigma(ii,ii) = .75;
    }

    return aux::GaussianPdf(mu, sigma);
}

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
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
//    RegularizedParticleResampler resampler_reg(NUM_PARTICLES);
  
    /* estimate and output results */
    aux::vector meas(BoldModel::MEAS_SIZE);
    aux::DiracMixturePdf pred(BoldModel::SYSTEM_SIZE);
    aux::vector mu(BoldModel::SYSTEM_SIZE);

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

    aux::vector sample_state(BoldModel::SYSTEM_SIZE);

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

