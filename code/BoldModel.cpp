#include "BoldModel.hpp"

#include <indii/ml/aux/matrix.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/DiracPdf.hpp>

#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>

#define EXPONENTIAL .2

BoldModel::BoldModel(bool expweight, bool avgweight, size_t sections, aux::vector u) :
            GVAR_SIZE(4), LVAR_SIZE(7), SIMUL_STATES(sections), 
            STATE_SIZE(GVAR_SIZE+LVAR_SIZE*SIMUL_STATES), MEAS_SIZE(SIMUL_STATES),
            INPUT_SIZE(1), segments(sections)
{
    //determines the layout of the state variables
    for(unsigned int i = 0 ; i<segments.size() ; i++) {
        segments[i].index = i*7;
    }

    //this is only a problem if the user put in a bad vector
    //in which case the u will be overwritten with 0's
    if(u.size() != sections)
        this->input = aux::zero_vector(sections);
    else 
        this->input = u;
    
    var_e = 3.92e-6;
    sigma_e = sqrt(var_e);

    if(expweight) 
        this->weightf = EXP;;

    if(avgweight)
        this->tweight = 5;

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
    for(unsigned int ii=0 ; ii<SIMUL_STATES ; ii++) {
        // Normalized Blood Volume
        //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
        dot1 = (  ( dustin[indexof(F_T,ii)] - 
                    pow(dustin[indexof(V_T,ii)], 1./dustin[indexof(ALPHA, ii)]) ) / 
                    dustin[indexof(TAU_0,ii)]  );
        dustin[indexof(V_T,ii)] += dot1*delta_t;
        
        if(isnan(dustin[indexof(V_T,ii)]) || isinf(dustin[indexof(V_T,ii)]) || 
                    dustin[indexof(V_T,ii)] < 0) {
            dustin[indexof(V_T,ii)] = 1;
            return -1;
        }

        // Normalized Deoxyhaemoglobin Content
        //Q_t* = \frac{1}{tau_0} * (\frac{f_t}{E_0} * (1- (1-E_0)^{1/f_t}) - 
        //              \frac{q_t}{v_t^{1-1/\alpha})
        tmpA = (dustin[indexof(F_T,ii)] / dustin[indexof(E_0, ii)]) * 
                    (1 - pow( 1. - dustin[indexof(E_0, ii)], 1./dustin[indexof(F_T,ii)]));
        tmpB = dustin[indexof(Q_T,ii)] / 
                    pow(dustin[indexof(V_T,ii)], 1.-1./dustin[indexof(ALPHA,ii)]);
        dot2 =  ( tmpA - tmpB )/dustin[indexof(TAU_0,ii)];
        dustin[indexof(Q_T,ii)] += dot2*delta_t;
        
        if(isnan(dustin[indexof(Q_T,ii)]) || isinf(dustin[indexof(Q_T,ii)]) || 
                    dustin[indexof(Q_T,ii)] < 0) {
            dustin[indexof(Q_T,ii)] = 1;
            return -2;
        }

        // Second Derivative of Cerebral Blood Flow
        //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
        dot3 = u_t[0]*dustin[indexof(EPSILON, ii)] 
                    - dustin[indexof(S_T,ii)]/dustin[indexof(TAU_S, ii)]
                    - (dustin[indexof(F_T,ii)] - 1.) / dustin[indexof(TAU_F,ii)];
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
    for(size_t i = 0 ; i < MEAS_SIZE ; i++) {
        
        y[i] = s[indexof(V_0, i)] * 
                ( A1 * ( 1 - s[indexof(Q_T,i)]) - A2 * (1 - s[indexof(V_T,i)]));
    }
    return y;
}

double BoldModel::weight(const aux::vector& s, const aux::vector& y)
{
    //these are really constant throughout the execution
    //of the program, so no need to calculate over and over
    static aux::symmetric_matrix cov(1);
    cov(0,0) = 1;
    static aux::GaussianPdf rng(aux::zero_vector(MEAS_SIZE), cov);
    aux::vector location(MEAS_SIZE);
//    fprintf(stderr, "Actual:\n");
//    outputVector(std::cerr, y);
//    fprintf(stderr, "Measure: \n");
//    outputVector(std::cerr, measure(s));
//    fprintf(stderr, "\nParticle:\n");
//    outputVector(std::cerr , s);
//    fprintf(stderr, "\n");
    
    //after computing the n-dimensional location whose density
    //will be found on a gaussian curve, set all terms that came
    //from y[i] = NaN1 elements to 0, thus eleminating them from the 
    //density calculation. The effectively ignores them for weighting
    //purposes, thus ignoring terms that received no update in this step.
    location = (y-measure(s))/sigma_e;
    for(size_t i = 0 ; i < y.size() ; i++) {
        if(isnan(y[i]))
            location[i] = 0;
    }
//    fprintf(stderr, "Location calculated:\n");
//    outputVector(std::cerr , location);
//    fprintf(stderr, "\n");
//    fprintf(stderr, "Weight calculated: %e\n", rng.densityAt(location));
//    return out;
    if(weightf == EXP) {
        return gsl_ran_exponential_pdf(aux::norm< 2 >(location), EXPONENTIAL);
    } else {
        return rng.densityAt(location);
    }
}


//Note that k_sigma contains std. deviation OR k and theta_mu contains either
//mean or theta depending on the distribution
void BoldModel::generate_component(gsl_rng* rng, aux::vector& fillme, 
            const double k_sigma[], const double theta_mu[]) 
{
    //going to distribute all the state variables the same even if they are
    //in different sections
    int count = 0;
    for(size_t i = 0 ; i< STATE_SIZE; i++) {
        if(indexof(S_T, count) == i) {
            //for S_t draw from a gaussian
            fillme[indexof(S_T, count)] = 
                        gsl_ran_gaussian(rng, k_sigma[indexof(S_T,count)])+
                        theta_mu[indexof(S_T,count)];
            count++;
        } else {
            //draw from the gama, assume independence between the variables
            fillme[i] = gsl_ran_gamma(rng, k_sigma[i], theta_mu[i]);
        }
    }
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples)
{
    aux::vector mean(STATE_SIZE);
    aux::symmetric_matrix cov = aux::zero_matrix(STATE_SIZE);
    
    //set the averages of the variables
    for(unsigned int ii = 0 ; ii < SIMUL_STATES; ii++) {
        mean[indexof(TAU_S, ii)] = 4.98;
        mean[indexof(TAU_F, ii)] = 8.31;
        mean[indexof(EPSILON, ii)] = 0.069;
        mean[indexof(TAU_0, ii)] = 8.38;
        mean[indexof(ALPHA, ii)] = .189;
        mean[indexof(E_0, ii)] = .635;
        mean[indexof(V_0, ii)] = 1.49e-2;

        mean[indexof(V_T,ii)] = 1;
        mean[indexof(Q_T,ii)] = 1;
        mean[indexof(S_T,ii)] = 0;
        mean[indexof(F_T,ii)]= 1;
        
        //set the variances for all the variables
        cov(indexof(TAU_S  ,ii), indexof(TAU_S  ,ii)) = 9*1.07*1.07;
        cov(indexof(TAU_F  ,ii), indexof(TAU_F  ,ii)) = 9*1.51*1.51;
        cov(indexof(EPSILON,ii), indexof(EPSILON,ii)) = 9*0.014*.014;
        cov(indexof(TAU_0  ,ii), indexof(TAU_0  ,ii)) = 9*1.5*1.5;
        cov(indexof(ALPHA  ,ii), indexof(ALPHA  ,ii)) = 9*.004*.004;
        cov(indexof(E_0    ,ii), indexof(E_0    ,ii)) = 9*.072*.072;
        cov(indexof(V_0    ,ii), indexof(V_0    ,ii)) = 9*.6e-2*.6e-2;

        cov(indexof(V_T,ii), indexof(V_T,ii)) = 2;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = 2;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = 2;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = 2;
    }
    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::vector mean)
{
    aux::symmetric_matrix cov = aux::zero_matrix(STATE_SIZE);
    
    for(unsigned int ii = 0 ; ii < SIMUL_STATES ; ii++) {
        //set the variances for all the variables to 3*sigma
        cov(indexof(TAU_S  ,ii), indexof(TAU_S  ,ii)) = 9*1.07*1.07;
        cov(indexof(TAU_F  ,ii), indexof(TAU_F  ,ii)) = 9*1.51*1.51;
        cov(indexof(EPSILON,ii), indexof(EPSILON,ii)) = 9*0.014*.014;
        cov(indexof(TAU_0  ,ii), indexof(TAU_0  ,ii)) = 9*1.5*1.5;
        cov(indexof(ALPHA  ,ii), indexof(ALPHA  ,ii)) = 9*.004*.004;
        cov(indexof(E_0    ,ii), indexof(E_0    ,ii)) = 9*.072*.072;
        cov(indexof(V_0    ,ii), indexof(V_0    ,ii)) = 9*.6e-2*.6e-2;

        cov(indexof(V_T,ii), indexof(V_T,ii)) = 2;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = 2;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = 2;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = 2;
    }
    
    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::symmetric_matrix cov)
{
    aux::vector mean(STATE_SIZE);
    for(unsigned int ii = 0 ; ii < SIMUL_STATES ; ii++) {
    //set the averages of the variables
        mean[indexof(TAU_S, ii)] = 4.98;
        mean[indexof(TAU_F, ii)] = 8.31;
        mean[indexof(EPSILON, ii)] = 0.069;
        mean[indexof(TAU_0, ii)] = 8.38;
        mean[indexof(ALPHA, ii)] = .189;
        mean[indexof(E_0, ii)] = .635;
        mean[indexof(V_0, ii)] = 1.49e-2;

        mean[indexof(V_T,ii)] = 1;
        mean[indexof(Q_T,ii)] = 1;
        mean[indexof(S_T,ii)] = 0;
        mean[indexof(F_T,ii)]= 1;
    }

    generatePrior(x0, samples, mean, cov);
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples,
            const aux::vector mean, const aux::symmetric_matrix cov)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();

    int count = 0;
    double k_sigma[STATE_SIZE]; 
    double theta_mu[STATE_SIZE];
    for(unsigned int i = 0 ; i < getStateSize() ; i++) {
        if(indexof(S_T, count) == i) {
            count++;
            theta_mu[i] = mean(i);
            k_sigma[i] = sqrt(cov(i,i));
        } else {
            theta_mu[i] = cov(i,i)/mean(i);
            k_sigma[i] = mean[i]/theta_mu[i];
        }
    }
    
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (int)((time(NULL)*(rank+11))/71.));
    aux::vector comp(STATE_SIZE);
    for(int i = 0 ; i < samples; i ++) {
        generate_component(rng, comp, k_sigma, theta_mu);
        x0.add(comp, 1.0);
    }
    gsl_rng_free(rng);
    }

//return weight modified?
bool BoldModel::reweight(aux::vector& checkme, double& weightout)
{
    size_t count = 0;
    for(unsigned int j = 0 ; j < checkme.size() ; j++) {
        //only S_T is allowed to be negative
        if(indexof(S_T, count) == j) {
            count++;
        } else if(checkme[j] < 0) {
//            for(unsigned int i = 0 ; i<checkme.size() ; i++)
//                checkme[i] = 1;
            weightout = 0.0;
            return true;
        } 
    }
    return false;
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

