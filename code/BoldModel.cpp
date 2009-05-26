#include "BoldModel.hpp"

#include <indii/ml/aux/matrix.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/DiracPdf.hpp>

#include <cmath>
#include <ctime>
#include <iomanip>

#define EXPONENTIAL .4

BoldModel::BoldModel(aux::vector u, int weightf, double tweight)
{
    if(THETA_SIZE + STATE_SIZE*SIMUL_STATES != SYSTEM_SIZE) {
        std::cerr << "Incorrect system size" << std::endl;
        exit(-1);
    }
    this->input = u;
//    theta_sigmas(TAU_S) = 1.07/20;
//    theta_sigmas(TAU_F) = 1.51/20;
//    theta_sigmas(EPSILON) = .014/20;
//    theta_sigmas(TAU_0) = 1.5/20;
//    theta_sigmas(ALPHA) = .004/20;
//    theta_sigmas(E_0) = .072/20;
//    theta_sigmas(V_0) = .006/20;

//    small_g = .95e-5;
    var_e = 3.92e-6;
    sigma_e = sqrt(var_e);

    this->weightf = weightf;
    this->tweight = tweight;
}

BoldModel::~BoldModel()
{

}

//TODO, I would like to modify these functions so that the vector s
//will just be modified in place, which would reduce the amount of copying
//necessary. This might not work though, because Particle Filter likes to
//keep all the particles around for history. After trying, this would mean
//that particle filter is no longer compatible with Filter.hpp.
int BoldModel::transition(aux::vector& s,
        const double t, const double delta)
{
    //use the default input
    return transition(s, t, delta, input);
}

//TODO make transition as FAST as possible
int BoldModel::transition(aux::vector& dustin,
        const double time, const double delta_t, const aux::vector& u_t)
{
//    std::cerr  <<"Printing input state" << std::endl;
//    outputVector(std::cerr, dustin);
//    std::cerr << std::endl;
    double dot1, dot2, dot3;
    double tmpA, tmpB;

    //transition the actual state variables
    //TODO, potentially add some randomness here.
    for(int ii=0 ; ii<SIMUL_STATES ; ii++) {
        // Normalized Blood Volume
        //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
        dot1 = (  ( dustin[indexof(F_T,ii)] - 
                    pow(dustin[indexof(V_T,ii)], 1./dustin[ALPHA]) ) / 
                    dustin[TAU_0]  );
        dustin[indexof(V_T,ii)] += dot1*delta_t;
        
        if(isnan(dustin[indexof(V_T,ii)]) || isinf(dustin[indexof(V_T,ii)]) || 
                    dustin[indexof(V_T,ii)] < 0) {
            dustin[indexof(V_T,ii)] = 1;
            return -1;
        }

        // Normalized Deoxyhaemoglobin Content
        //Q_t* = \frac{1}{tau_0} * (\frac{f_t}{E_0} * (1- (1-E_0)^{1/f_t}) - 
        //              \frac{q_t}{v_t^{1-1/\alpha})
        tmpA = (dustin[indexof(F_T,ii)] / dustin[E_0]) * 
                    (1 - pow( 1. - dustin[E_0], 1./dustin[indexof(F_T,ii)]));
        tmpB = dustin[indexof(Q_T,ii)] / 
                    pow(dustin[indexof(V_T,ii)], 1.-1./dustin[ALPHA]);
        dot2 =  ( tmpA - tmpB )/dustin[TAU_0];
        dustin[indexof(Q_T,ii)] += dot2*delta_t;
        
        if(isnan(dustin[indexof(Q_T,ii)]) || isinf(dustin[indexof(Q_T,ii)]) || 
                    dustin[indexof(Q_T,ii)] < 0) {
            dustin[indexof(Q_T,ii)] = 1;
            return -2;
        }

        // Second Derivative of Cerebral Blood Flow
        //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
        dot3 = u_t[0]*dustin[EPSILON]- dustin[indexof(S_T,ii)]/dustin[TAU_S] - 
                    (dustin[indexof(F_T,ii)] - 1.) / dustin[TAU_F];
        dustin[indexof(S_T,ii)] += dot3*delta_t;
       
        // Normalized Cerebral Blood Flow
        //f_t* = s_t;
        dustin[indexof(F_T,ii)] += dustin[indexof(S_T,ii)]*delta_t;
        
        if(dustin[indexof(F_T,ii)] < 0) {
            dustin[indexof(F_T,ii)] = 1;
            return -3;
        }
    }
        
//    std::cerr  <<"Printing Output state" << std::endl;
//    outputVector(std::cerr, dustout);
//    std::cerr << std::endl;
    return 0;
}

aux::vector BoldModel::measure(const aux::vector& s)
{
    aux::vector y(MEAS_SIZE);
    for(int i = 0 ; i < MEAS_SIZE ; i++) {
        y[i] = s[V_0] * ( A1 * ( 1 - s[indexof(Q_T,i)]) - A2 * (1 - s[indexof(V_T,i)]));
    }
    return y;
}

double BoldModel::weight(const aux::vector& s, const aux::vector& y)
{
    //these are really constant throughout the execution
    //of the program, so no need to calculate over and over
    static aux::symmetric_matrix cov(1);
    cov(0,0) = 1;
#ifndef EXPONENTIAL
    static aux::GaussianPdf rng(aux::zero_vector(MEAS_SIZE), cov);
#endif 
    aux::vector location(MEAS_SIZE);
//    fprintf(stderr, "Actual:\n");
//    outputVector(std::cerr, y);
//    fprintf(stderr, "Measure: \n");
//    outputVector(std::cerr, measure(s));
//    fprintf(stderr, "\nParticle:\n");
//    outputVector(std::cerr , s);
//    fprintf(stderr, "\n");
    
    location = (y-measure(s))/sigma_e;
//    fprintf(stderr, "Location calculated:\n");
//    outputVector(std::cerr , location);
//    fprintf(stderr, "\n");
//    fprintf(stderr, "Weight calculated: %e\n", rng.densityAt(location));
//    return out;
#ifdef EXPONENTIAL
//  use exponential distribution with mean = lambda = .01
    return gsl_ran_exponential_pdf(fabs(location(0)), EXPONENTIAL);
#else
    return rng.densityAt(location);
#endif
}


//Note that k_sigma contains std. deviation OR k and theta_mu contains either
//mean or theta depending on the distribution
void BoldModel::generate_component(gsl_rng* rng, aux::vector& fillme, 
            const double k_sigma[], const double theta_mu[]) 
{
    //draw from the gama, assume independence between the variables

    fillme[TAU_S]   = gsl_ran_gamma(rng, k_sigma[TAU_S],   theta_mu[TAU_S]);
    fillme[TAU_F]   = gsl_ran_gamma(rng, k_sigma[TAU_F],   theta_mu[TAU_F]);
    fillme[EPSILON] = gsl_ran_gamma(rng, k_sigma[EPSILON], theta_mu[EPSILON]);
    fillme[TAU_0]   = gsl_ran_gamma(rng, k_sigma[TAU_0],   theta_mu[TAU_0]);
    fillme[ALPHA]   = gsl_ran_gamma(rng, k_sigma[ALPHA],   theta_mu[ALPHA]);
    fillme[E_0]     = gsl_ran_gamma(rng, k_sigma[E_0],     theta_mu[E_0]);
    fillme[V_0]     = gsl_ran_gamma(rng, k_sigma[V_0],     theta_mu[V_0]);

    //going to distribute all the state variables the same even if they are
    //in different sections
    for(int i = 0 ; i< SIMUL_STATES ; i++) {
        fillme[indexof(V_T, i)] = gsl_ran_gamma(rng, k_sigma[indexof(V_T,0)], 
                    theta_mu[indexof(V_T,0)]);
        fillme[indexof(Q_T, i)] = gsl_ran_gamma(rng, k_sigma[indexof(Q_T,0)], 
                    theta_mu[indexof(Q_T,0)]);
        fillme[indexof(S_T, i)] = gsl_ran_gaussian(rng, k_sigma[indexof(S_T,0)])+
                    theta_mu[indexof(S_T,0)];
        fillme[indexof(F_T, i)] = gsl_ran_gamma(rng, k_sigma[indexof(F_T,0)], 
                    theta_mu[indexof(F_T,0)]);
    }
    
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();

    aux::vector mean(SYSTEM_SIZE);
    aux::symmetric_matrix cov = aux::zero_matrix(SYSTEM_SIZE);
    
    //set the averages of the variables
    mean[0] = 4.98;
    mean[1] = 8.31;
    mean[2] = 0.069;
    mean[3] = 8.38;
    mean[4] = .189;
    mean[5] = .635;
    mean[6] = 1.49e-2;

    mean[7] = 1;
    mean[8] = 1;
    mean[9] = 0;
    mean[10]= 1;
    
    //set the variances for all the variables
    cov(0,0) = 4*1.07*1.07;
    cov(1,1) = 4*1.51*1.51;
    cov(2,2) = 4*0.014*.014;
    cov(3,3) = 4*1.5*1.5;
    cov(4,4) = 4*.004*.004;
    cov(5,5) = 4*.072*.072;
    cov(6,6) = 4*.6e-2*.6e-2;

    cov(7,7) = .5;
    cov(8,8) = .5;
    cov(9,9) = .5;
    cov(10,10) = .5;

    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::vector mean)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();

    aux::symmetric_matrix cov = aux::zero_matrix(SYSTEM_SIZE);
    
    //set the variances for all the variables
    cov(0,0) = 4*1.07*1.07;
    cov(1,1) = 4*1.51*1.51;
    cov(2,2) = 4*0.014*.014;
    cov(3,3) = 4*1.5*1.5;
    cov(4,4) = 4*.004*.004;
    cov(5,5) = 4*.072*.072;
    cov(6,6) = 4*.6e-2*.6e-2;

    cov(7,7) = .5;
    cov(8,8) = .5;
    cov(9,9) = .5;
    cov(10,10) = .5;

    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::symmetric_matrix cov)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();

    aux::vector mean(SYSTEM_SIZE);
    
    //set the averages of the variables
    mean[0] = 4.98;
    mean[1] = 8.31;
    mean[2] = 0.069;
    mean[3] = 8.38;
    mean[4] = .189;
    mean[5] = .635;
    mean[6] = 1.49e-2;

    mean[7] = 1;
    mean[8] = 1;
    mean[9] = 0;
    mean[10]= 1;

    generatePrior(x0, samples, mean, cov);
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples,
            const aux::vector mean, const aux::symmetric_matrix cov)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    
    //set the averages of the variables
    double mu_TAU_S = 4.98;
    double mu_TAU_F = 8.31;
    double mu_EPSILON = 0.069;
    double mu_TAU_0 = 8.38;
    double mu_ALPHA = .189;
    double mu_E_0 = .635;
    double mu_V_0 = 1.49e-2;
    
    double mu_V_T = 1;
    double mu_Q_T = 1;
    double mu_S_T = 0;
    double mu_F_T = 1;
    
    //set the variances for all the variables
    double var_TAU_S =   4*1.07*1.07;
    double var_TAU_F =   4*1.51*1.51;
    double var_EPSILON = 4*0.014*.014;
    double var_TAU_0 =   4*1.5*1.5;
    double var_ALPHA =   4*.004*.004;
    double var_E_0 =     4*.072*.072;
    double var_V_0 =     4*.6e-2*.6e-2;
    
    double var_V_T = .5;
    double var_Q_T = .5;
    double var_S_T = .5;
    double var_F_T = .1;
    
    //set the theta of the variables
    double theta_TAU_S   = var_TAU_S/mu_TAU_S;
    double theta_TAU_F   = var_TAU_F/mu_TAU_F;
    double theta_EPSILON = var_EPSILON/mu_EPSILON;
    double theta_TAU_0   = var_TAU_0/mu_TAU_0;
    double theta_ALPHA   = var_ALPHA/mu_ALPHA;
    double theta_E_0     = var_E_0/mu_E_0;
    double theta_V_0     = var_V_0/mu_V_0;
    
    double theta_V_T = var_V_T/mu_V_T;
    double theta_Q_T = var_Q_T/mu_Q_T;
    double theta_F_T = var_F_T/mu_F_T;

    //set the k of the variables
    double k_TAU_S   = mu_TAU_S/theta_TAU_S;
    double k_TAU_F   = mu_TAU_F/theta_TAU_F;
    double k_EPSILON = mu_EPSILON/theta_EPSILON;
    double k_TAU_0   = mu_TAU_0/theta_TAU_0;
    double k_ALPHA   = mu_ALPHA/theta_ALPHA;
    double k_E_0     = mu_E_0/theta_E_0;
    double k_V_0     = mu_V_0/theta_V_0;
    
    double k_V_T = mu_V_T/theta_V_T;
    double k_Q_T = mu_Q_T/theta_Q_T;
    double sigma_S_T = sqrt(var_S_T);
    double k_F_T = mu_F_T/theta_F_T;


    double k_sigma[] = {k_TAU_S, k_TAU_F, k_EPSILON, k_TAU_0, k_ALPHA, k_E_0,
                k_V_0, k_V_T, k_Q_T, sigma_S_T, k_F_T};
    double theta_mu[] = {theta_TAU_S, theta_TAU_F, theta_EPSILON, 
                theta_TAU_0, theta_ALPHA, theta_E_0, theta_V_0, theta_V_T, 
                theta_Q_T, mu_S_T, theta_F_T};

    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (int)((time(NULL)*(rank+11))/71.));
    aux::vector comp(SYSTEM_SIZE);
    for(int i = 0 ; i < samples; i ++) {
        generate_component(rng, comp, k_sigma, theta_mu);
        x0.add(comp, 1.0);
    }
    gsl_rng_free(rng);
}

void outputVector(std::ostream& out, aux::vector mat) {
  unsigned int i;
  for (i = 0; i < mat.size(); i++) {
      out << std::setw(15) << mat(i);
  }
}

void outputMatrix(std::ostream& out, aux::matrix mat) {
  unsigned int i, j;
  for (j = 0; j < mat.size2(); j++) {
    for (i = 0; i < mat.size1(); i++) {
      out << std::setw(15) << mat(i,j);
    }
    out << std::endl;
  }
}
