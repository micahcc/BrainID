#include "BoldModel.hpp"

#include <indii/ml/aux/matrix.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/DiracPdf.hpp>

#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>

#define EXPONENTIAL_VAR .2
#define GAUSSIAN_VAR .2

#include <iostream>

BoldModel::BoldModel(bool expweight, size_t sections, double var,
            aux::vector u) : 

            GVAR_SIZE(4), LVAR_SIZE(7), SIMUL_STATES(sections), 
            STATE_SIZE(GVAR_SIZE+LVAR_SIZE*SIMUL_STATES), MEAS_SIZE(SIMUL_STATES),
            INPUT_SIZE(1)//, segments(sections)
{
//    //determines the layout of the state variables
//    for(unsigned int i = 0 ; i<segments.size() ; i++) {
//        segments[i].index = i*7;
//    }

    //this is only a problem if the user put in a bad vector
    //in which case the u will be overwritten with 0's
    if(u.size() != sections)
        this->input = aux::zero_vector(sections);
    else 
        this->input = u;
    
    this->var_e = var;
    this->sigma_e = sqrt(var);

    if(expweight) 
        this->weightf = EXP;;

}

BoldModel::~BoldModel()
{

}

aux::vector BoldModel::getdefault()
{
    aux::vector fillme(STATE_SIZE);
    //set the averages of the variables
    for(unsigned int ii = 0 ; ii < SIMUL_STATES; ii++) {
        fillme[indexof(TAU_S, ii)] = 4.98;
        fillme[indexof(TAU_F, ii)] = 8.31;
        fillme[indexof(EPSILON, ii)] = 0.069;
        fillme[indexof(TAU_0, ii)] = 8.38;
        fillme[indexof(ALPHA, ii)] = .189;
        fillme[indexof(E_0, ii)] = .635;
        fillme[indexof(V_0, ii)] = 1.49e-2;

        fillme[indexof(V_T,ii)] = 1;
        fillme[indexof(Q_T,ii)] = 1;
        fillme[indexof(S_T,ii)] = 0;
        fillme[indexof(F_T,ii)]= 1;
    }
    return fillme;
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
    static aux::vector defaultvector = getdefault();
//    std::cerr  <<"Printing input state" << std::endl;
//    outputVector(std::cerr, dustin);
//    std::cerr << std::endl;
    double dot1, dot2, dot3;
    double tmpA, tmpB;

    //transition the actual state variables
    //TODO, potentially add some randomness here.
    for(unsigned int ii=0 ; ii<SIMUL_STATES ; ii++) {
        unsigned int v_t = indexof(V_T,ii);
        unsigned int q_t = indexof(Q_T,ii);
        unsigned int s_t = indexof(S_T,ii);
        unsigned int f_t = indexof(F_T,ii);
        // Normalized Blood Volume
        //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
        dot1 = (  ( dustin[f_t] - 
                    pow(dustin[v_t], 1./dustin[indexof(ALPHA, ii)]) ) / 
                    dustin[indexof(TAU_0,ii)]  );

        // Normalized Deoxyhaemoglobin Content
        //Q_t* = \frac{1}{tau_0} * (\frac{f_t}{E_0} * (1- (1-E_0)^{1/f_t}) - 
        //              \frac{q_t}{v_t^{1-1/\alpha})
        tmpA = (dustin[f_t] / dustin[indexof(E_0, ii)]) * 
                    (1 - pow( 1. - dustin[indexof(E_0, ii)], 1./dustin[f_t]));
        tmpB = dustin[q_t] / 
                    pow(dustin[v_t], 1.-1./dustin[indexof(ALPHA,ii)]);
        dot2 =  ( tmpA - tmpB )/dustin[indexof(TAU_0,ii)];

        // Second Derivative of Cerebral Blood Flow
        //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
        dot3 = u_t[0]*dustin[indexof(EPSILON, ii)] 
                    - dustin[s_t]/dustin[indexof(TAU_S, ii)]
                    - (dustin[f_t] - 1.) / dustin[indexof(TAU_F,ii)];
        // Normalized Cerebral Blood Flow
        //f_t* = s_t;
        dustin[f_t] += dustin[s_t]*delta_t;
        if(dustin[f_t] < 0 || isnan(dustin[f_t]) || isinf(dustin[f_t]) ) {
            dustin = defaultvector;
            return -4;
        }

        /* Update the others based on their gradient */
        dustin[v_t] += dot1*delta_t;
        if(isnan(dustin[v_t]) || isinf(dustin[v_t]) || dustin[v_t] < 0) {
            dustin = defaultvector;
            return -1;
        }
        
        dustin[q_t] += dot2*delta_t;
        if(isnan(dustin[q_t]) || isinf(dustin[q_t]) || dustin[q_t] < 0) {
            dustin = defaultvector;
            return -2;
        }
        
        dustin[s_t] += dot3*delta_t;
        if(isnan(dustin[s_t]) || isinf(dustin[s_t])) {
            dustin = defaultvector;
            return -3;
        }
       
    }
        
//    std::cerr  <<"Printing Output state" << std::endl;
//    outputVector(std::cerr, dustin);
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
    location = y-measure(s);
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
        return gsl_ran_exponential_pdf(aux::norm< 2 >(location), sigma_e);
    } else {
        return gsl_ran_gaussian_pdf(aux::norm< 2 >(location), sigma_e);
    }
}


//Note that k_sigma contains std. deviation OR k and theta_mu contains either
//mean or theta depending on the distribution
void BoldModel::generate_component(gsl_rng* rng, aux::vector& fillme, 
            const double* k_sigma, const double* theta_mu) 
{
    //going to distribute all the state variables the same even if they are
    //in different sections
    int count = 0;
    for(size_t i = 0 ; i< STATE_SIZE; i++) {
//            fillme[i] = gsl_ran_gaussian(rng, k_sigma[i])+ theta_mu[i];
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
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, double varwidth)
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
        cov(indexof(TAU_S  ,ii), indexof(TAU_S  ,ii)) = varwidth*1.07*1.07;
        cov(indexof(TAU_F  ,ii), indexof(TAU_F  ,ii)) = varwidth*1.51*1.51;
        cov(indexof(EPSILON,ii), indexof(EPSILON,ii)) = varwidth*0.014*.014;
        cov(indexof(TAU_0  ,ii), indexof(TAU_0  ,ii)) = varwidth*1.5*1.5;
        cov(indexof(ALPHA  ,ii), indexof(ALPHA  ,ii)) = varwidth*.004*.004;
        cov(indexof(E_0    ,ii), indexof(E_0    ,ii)) = varwidth*.072*.072;
        cov(indexof(V_0    ,ii), indexof(V_0    ,ii)) = varwidth*.6e-2*.6e-2;

        cov(indexof(V_T,ii), indexof(V_T,ii)) = varwidth*.2;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = varwidth*.2;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = varwidth*.6;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = varwidth*1;
    }
    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::vector mean, double varwidth)
{
    aux::symmetric_matrix cov = aux::zero_matrix(STATE_SIZE);
    
    for(unsigned int ii = 0 ; ii < SIMUL_STATES ; ii++) {
        //set the variances for all the variables to 3*sigma
        cov(indexof(TAU_S  ,ii), indexof(TAU_S  ,ii)) = varwidth*1.07*1.07;
        cov(indexof(TAU_F  ,ii), indexof(TAU_F  ,ii)) = varwidth*1.51*1.51;
        cov(indexof(EPSILON,ii), indexof(EPSILON,ii)) = varwidth*0.014*.014;
        cov(indexof(TAU_0  ,ii), indexof(TAU_0  ,ii)) = varwidth*1.5*1.5;
        cov(indexof(ALPHA  ,ii), indexof(ALPHA  ,ii)) = varwidth*.004*.004;
        cov(indexof(E_0    ,ii), indexof(E_0    ,ii)) = varwidth*.072*.072;
        cov(indexof(V_0    ,ii), indexof(V_0    ,ii)) = varwidth*.6e-2*.6e-2;

        cov(indexof(V_T,ii), indexof(V_T,ii)) = varwidth*.2;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = varwidth*.2;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = varwidth*.6;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = varwidth*1;
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
    static aux::vector defaultvector = getdefault();
    size_t count = 0;
    for(unsigned int j = 0 ; j < checkme.size() ; j++) {
        //only S_T is allowed to be negative
        if(indexof(S_T, count) == j) {
            count++;
        } else if(checkme[j] < 0) {
//            for(unsigned int i = 0 ; i<checkme.size() ; i++)
//                checkme[i] = 1;
            weightout = 0.0;
            checkme = defaultvector;
            return true;
        } 
    }
    return false;
}

