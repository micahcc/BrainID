#include "BoldModel.hpp"
#include "tools.h"

#include <indii/ml/aux/matrix.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/DiracPdf.hpp>

#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>

BoldModel::BoldModel(aux::vector stddev, int weightfunc, 
            size_t sections, aux::vector drift) : 
            weightf(weightfunc), sigma(stddev), SIMUL_STATES(sections),
            STATE_SIZE(GVAR_SIZE+LVAR_SIZE*sections + sections),
            MEAS_SIZE(sections),
            INPUT_SIZE(1)//, segments(sections)
{
    //this is only a problem if the user put in a bad vector
    //in which case the u will be overwritten with 0's
    this->input = aux::zero_vector(sections);
    
    defaultstate.resize(STATE_SIZE, false);
    //set the averages of the variables
    for(unsigned int ii = 0 ; ii < SIMUL_STATES; ii++) {
        defaultstate[indexof(TAU_S, ii)] = 4.98;
        defaultstate[indexof(TAU_F, ii)] = 8.31;
        defaultstate[indexof(EPSILON, ii)] = 0.069;
        defaultstate[indexof(TAU_0, ii)] = 8.38;
        defaultstate[indexof(ALPHA, ii)] = .189;
        defaultstate[indexof(E_0, ii)] = .635;
        defaultstate[indexof(V_0, ii)] = 1.49e-2;

        defaultstate[indexof(V_T,ii)] = 1;
        defaultstate[indexof(Q_T,ii)] = 1;
        defaultstate[indexof(S_T,ii)] = 0;
        defaultstate[indexof(F_T,ii)]= 1;
    }
    
    if(drift.size() != MEAS_SIZE) 
        drift = aux::vector(MEAS_SIZE, 0);
    for(unsigned int i = 0; i < MEAS_SIZE ; i++)
        defaultstate[STATE_SIZE-MEAS_SIZE+i] = drift[i];

    std::cerr << "Sizes:" << std::endl;
    std::cerr << "GVAR_SIZE    " << this->GVAR_SIZE    << std::endl;
    std::cerr << "LVAR_SIZE    " << this->LVAR_SIZE    << std::endl;
    std::cerr << "SIMUL_STATES " << this->SIMUL_STATES << std::endl;
    std::cerr << "STATE_SIZE   " << this->STATE_SIZE   << std::endl;
    std::cerr << "MEAS_SIZE    " << this->MEAS_SIZE    << std::endl;
    std::cerr << "INPUT_SIZE   " << this->INPUT_SIZE   << std::endl;

    for(unsigned int i = 0 ; i < 3 ; i++) {
        std::cerr << "indexof(TAU_0, " << i << ") " << indexof(TAU_0, i) << std::endl;
        std::cerr << "indexof(ALPHA, " << i << ") " << indexof(ALPHA, i) << std::endl;
        std::cerr << "indexof(E_0  , " << i << ") " << indexof(E_0  , i) << std::endl;
        std::cerr << "indexof(V_0  , " << i << ") " << indexof(V_0  , i) << std::endl;
        std::cerr << "indexof(TAU_S, " << i << ") " << indexof(TAU_S, i) << std::endl;
        std::cerr << "indexof(TAU_F, " << i << ") " << indexof(TAU_F, i) << std::endl;
        std::cerr << "indexof(EPSILON, " << i << ") " << indexof(EPSILON, i) << std::endl;
        std::cerr << "indexof(V_T  , " << i << ") " << indexof(V_T  , i) << std::endl;
        std::cerr << "indexof(Q_T  , " << i << ") " << indexof(Q_T  , i) << std::endl;
        std::cerr << "indexof(S_T  , " << i << ") " << indexof(S_T  , i) << std::endl;
        std::cerr << "indexof(F_T  , " << i << ") " << indexof(F_T  , i) << std::endl;
    }
    for(unsigned int i = 0 ; i < MEAS_SIZE; i++) {
        std::cerr << "Extra vars: " << i << " -> " << STATE_SIZE-MEAS_SIZE+i;
    }

    std::cerr << "Sigma:" << std::endl;
    for(unsigned int i = 0 ; i < this->sigma.size() ; i++)
        std::cerr << this->sigma(i) << std::endl;
}

BoldModel::~BoldModel()
{

}

int BoldModel::transition(aux::vector& s, const double t, const double delta) const
{
    //use the default input
    return transition(s, t, delta, input);
}

//TODO make transition as FAST as possible
int BoldModel::transition(aux::vector& dustin, const double time, 
            const double delta_t, const aux::vector& u_t) const
{
    static aux::vector defaultvector = getdefault();
    double dotV, dotQ, dotS;
    double tmpA, tmpB;
    double tmp = 0;

    //transition the actual state variables
    for(unsigned int ii=0 ; ii<SIMUL_STATES ; ii++) {
        unsigned int v_t = indexof(V_T,ii);
        unsigned int q_t = indexof(Q_T,ii);
        unsigned int s_t = indexof(S_T,ii);
        unsigned int f_t = indexof(F_T,ii);
        // Normalized Blood Volume
        //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
        dotV = (  ( dustin[f_t] - 
                    pow(dustin[v_t], 1./dustin[indexof(ALPHA, ii)]) ) / 
                    dustin[indexof(TAU_0,ii)]  );

        // Normalized Deoxyhaemoglobin Content
        //Q_t* = \frac{1}{tau_0} * (\frac{f_t}{E_0} * (1- (1-E_0)^{1/f_t}) - 
        //              \frac{q_t}{v_t^{1-1/\alpha})
        tmpA = (dustin[f_t] / dustin[indexof(E_0, ii)]) * 
                    (1 - pow( 1. - dustin[indexof(E_0, ii)], 1./dustin[f_t]));
        tmpB = dustin[q_t] / 
                    pow(dustin[v_t], 1.-1./dustin[indexof(ALPHA,ii)]);
        dotQ =  ( tmpA - tmpB )/dustin[indexof(TAU_0,ii)];

        // Second Derivative of Cerebral Blood Flow
        //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
        dotS = u_t[0]*dustin[indexof(EPSILON, ii)] 
                    - dustin[s_t]/dustin[indexof(TAU_S, ii)]
                    - (dustin[f_t] - 1.) / dustin[indexof(TAU_F,ii)];
        // Normalized Cerebral Blood Flow
        //f_t* = s_t;
        dustin[f_t] += dustin[s_t]*delta_t;
        dustin[v_t] += dotV*delta_t;
        dustin[q_t] += dotQ*delta_t;
        dustin[s_t] += dotS*delta_t;
        tmp += dustin[f_t] + dustin[v_t] + dustin[q_t] + dustin[s_t];
    }

    /* Check for Nan, (nan operated with anything is nan),
     * inf - inf = nan, so nan or inf in any member 
     * will cause this to fail
    */
    if(isnan(tmp) || isinf(tmp)) {
        dustin = defaultvector;
        return -1;
    }
        
    return 0;
}

aux::vector BoldModel::measure(const aux::vector& s) const
{
    aux::vector y(MEAS_SIZE);
    for(size_t i = 0 ; i < MEAS_SIZE ; i++) {
        y[i] = s[indexof(V_0, i)] * 
                ( A1 * ( 1 - s[indexof(Q_T,i)]) - A2 * (1 - s[indexof(V_T,i)]));
    }
    return y;
} 
/* If s[STATE_SIZE-MEAS_SIZE+1] is negative, then this is working
 * with a "drift" variable, aka an arbitrary number is being ADDED
 * to the measurement to account for low frequency drift. In this case
 * y should be the RAW Bold signal
 * The drift variable will adapt in due course of the regularized particle
 * filter, as long as there was some variance in the initial distribution of
 * these values
 *
 * If s[STATE_SIZE-MEAS_SIZE+1] is positive, then it should be a difference
 * term from a previous measurement, in which case y should be the delta
 * in the Bold from a previous measurement. It is up to the user to make
 * sure those s values are correctly inserted.
 *
 * If s[STATE_SIZE-MEAS_SIZE+1] is 0, then y should be from a signal
 * with drift removed by some extra-pf method. The variance in these
 * drift terms will be 0, which will result in no variance from the
 * regularized particle filter. THus they will stay 0 perpetually.
 *
*/
double BoldModel::weight(const aux::vector& s, const aux::vector& y) const
{
    aux::vector meas = measure(s);
    double weight = 1;
    switch (weightf) {
        /* Cauchyt Distribution */
        case CAUCHY:
    	for(unsigned int i = 0 ; i < MEAS_SIZE ; i++)
            weight *= gsl_ran_cauchy_pdf( y[i]-meas[i]+
                        s[STATE_SIZE-MEAS_SIZE+i], sigma(i));
        break;
        /* Hyperbolic "Distribution" */
        case HYP:
    	for(unsigned int i = 0 ; i < MEAS_SIZE ; i++)
            weight *= sigma(i)/(y[i]-meas[i]+
                        s[STATE_SIZE-MEAS_SIZE+i]);
        break;
        /* Laplace Distribution */
        case LAPLACE:
    	for(unsigned int i = 0 ; i < MEAS_SIZE ; i++)
            weight *= gsl_ran_laplace_pdf( y[i]-meas[i]+
                        s[STATE_SIZE-MEAS_SIZE+i], sigma(i));
        break;
        /* Normal Distribution (default) */
        case NORM:
    	for(unsigned int i = 0 ; i < MEAS_SIZE ; i++)
            weight *= gsl_ran_gaussian_pdf( y[i]-meas[i]+
                        s[STATE_SIZE-MEAS_SIZE+i], sigma(i));
        default:
        break;
    }
    return fabs(weight);
}


//Note that scale_p contains std. deviation OR k and loc_p contains either
//mean or theta depending on the distribution
double BoldModel::generateComponent(gsl_rng* rng, aux::vector& fillme, 
            const double* scale_p, const double* loc_p) const
{
    double weight = 1, tmp;
    //going to distribute all the state variables the same even if they are
    //in different sections
    int count = 0;

    for(size_t i = 0 ; i< STATE_SIZE - MEAS_SIZE; i++) {
        if(scale_p[i] == 0) {
            fillme[i] = loc_p[i];
        } else if(indexof(S_T, count) == i) {
            //for S_t draw from a gaussian
            tmp = gsl_ran_gaussian(rng, scale_p[i]);
            fillme[i] = tmp + loc_p[i];
            weight *= gsl_ran_gaussian_pdf(tmp, scale_p[i]);
            count++;
        } else {
            //draw from the gamma, assume independence between the variables
            fillme[i] = gsl_ran_gamma(rng, loc_p[i], scale_p[i]);
            weight *= gsl_ran_gamma_pdf(fillme[i], scale_p[i], loc_p[i]);
        }
    }
    
    for(size_t i = STATE_SIZE - MEAS_SIZE; i < STATE_SIZE; i++) {
        if(scale_p[i] < 1e-10) {
            fillme[i] = loc_p[i];
        } else {
            tmp = gsl_ran_gaussian(rng, scale_p[i]);
            fillme[i] = tmp + loc_p[i];
            weight *= gsl_ran_gaussian_pdf(tmp, scale_p[i]);
        }
    }
    return 1./weight;
}
    
//approximated the variance by assuming V_0 is a constant, rather than
//random, because the variance of V_0 is actually pretty small
aux::vector BoldModel::estMeasVar(aux::DiracMixturePdf& in) const
{
    boost::mpi::communicator world;
    aux::vector var(MEAS_SIZE);
    aux::matrix cov = in.getDistributedCovariance();
    aux::vector exp = in.getDistributedExpectation();
    double v0sqr = 0;
    for(unsigned int i = 0 ; i < var.size() ; i++) {
        v0sqr = exp[indexof(V_0, i)] * exp[indexof(V_0, i)];
        var[i] = A1*A1*v0sqr*cov(indexof(Q_T, i), indexof(Q_T, i))+ 
                    A2*A2*v0sqr*cov(indexof(V_T, i), indexof(V_T, i))-
                    2*A1*A2*v0sqr*cov(indexof(V_T, i), indexof(Q_T, i));
    }
    return var;
}

aux::vector BoldModel::estMeasMean(aux::DiracMixturePdf& in) const
{
    return measure(in.getDistributedExpectation());
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            double varwidth, bool flat) const
{
    aux::vector mean = getdefault();
    aux::symmetric_matrix cov = aux::zero_matrix(STATE_SIZE);
    
    //set the averages of the variables
    for(unsigned int ii = 0 ; ii < SIMUL_STATES; ii++) {
        //set the variances for all the variables
        cov(indexof(TAU_S  ,ii), indexof(TAU_S  ,ii)) = varwidth*1.07*1.07;
        cov(indexof(TAU_F  ,ii), indexof(TAU_F  ,ii)) = varwidth*1.51*1.51;
        cov(indexof(EPSILON,ii), indexof(EPSILON,ii)) = varwidth*0.014*.014;
        cov(indexof(TAU_0  ,ii), indexof(TAU_0  ,ii)) = varwidth*1.5*1.5;
        cov(indexof(ALPHA  ,ii), indexof(ALPHA  ,ii)) = varwidth*.004*.004;
        cov(indexof(E_0    ,ii), indexof(E_0    ,ii)) = varwidth*.072*.072;
        cov(indexof(V_0    ,ii), indexof(V_0    ,ii)) = varwidth*.6e-2*.6e-2;

        cov(indexof(V_T,ii), indexof(V_T,ii)) = 1e-20;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = 1e-20;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = 1e-20;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = 1e-20;
    }

    for(unsigned int ii = STATE_SIZE-MEAS_SIZE ; ii < STATE_SIZE; ii++) {
        cov(ii,ii) = pow(sigma[ii-STATE_SIZE+MEAS_SIZE], 2);
    }
    generatePrior(x0, samples, mean, cov, flat);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::vector mean, double varwidth, bool flat) const
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

        //Assume they start at 0
        cov(indexof(V_T,ii), indexof(V_T,ii)) = 1e-20;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = 1e-20;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = 1e-20;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = 1e-20;
    }
    for(unsigned int ii = STATE_SIZE-MEAS_SIZE ; ii < STATE_SIZE; ii++) {
        cov(ii,ii) = pow(sigma[ii-STATE_SIZE+MEAS_SIZE], 2);
    }
    
    generatePrior(x0, samples, mean, cov, flat);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::symmetric_matrix cov, bool flat) const
{
    aux::vector mean = getdefault();
    generatePrior(x0, samples, mean, cov, flat);
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples,
            const aux::vector mean, const aux::symmetric_matrix cov, bool flat) const
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    
    unsigned int count = 0;
    double scale_p[STATE_SIZE]; 
    double loc_p[STATE_SIZE];

    for(unsigned int i = 0 ; i < STATE_SIZE-MEAS_SIZE; i++) {
        if(indexof(S_T, count) == i) {
            count++;
            loc_p[i] = mean(i);
            scale_p[i] = sqrt(cov(i,i));
        } else if(cov(i,i) > 1e-10) {
            scale_p[i] = (-mean[i]+sqrt(mean[i]*mean[i]+4*cov(i,i)))/2; //theta
            loc_p[i] = cov(i,i)/(scale_p[i]*scale_p[i]); //K
        } else {
            scale_p[i] = 0;
            loc_p[i] = mean[i];
        }
    }
    
    for(unsigned int i = STATE_SIZE-MEAS_SIZE; i < STATE_SIZE ; i++) {
        loc_p[i] = mean(i);
        scale_p[i] = sqrt(cov(i,i));
    }

    gsl_rng* rng = gsl_rng_alloc(gsl_rng_ran3);
    {
        unsigned int seed;
        FILE* file = fopen("/dev/urandom", "r");
        fread(&seed, 1, sizeof(unsigned int), file);
        fclose(file);
        gsl_rng_set(rng, seed^rank);
        std::cout << "Seeding with " << (unsigned int)(seed^rank) << "\n";
    }
    std::cout << "scale_p: ";
    for(unsigned int i = 0 ; i < STATE_SIZE ; i++)
        std::cout << " " << scale_p[i];
    std::cout << "\nloc_p: ";
    for(unsigned int i = 0 ; i < STATE_SIZE ; i++)
        std::cout << " " << loc_p[i];
    std::cout <<  "\n";
    aux::vector comp(STATE_SIZE);
    double weight;
    for(int i = 0 ; i < samples; i ++) {
        weight = generateComponent(rng, comp, scale_p, loc_p);
        x0.add(comp, flat ? weight : 1.0);
    }
    gsl_rng_free(rng);
}

//return (weight modified)
bool BoldModel::reweight(aux::vector& checkme, double& weightout) const
{
    static aux::vector defaultvector = getdefault();
    size_t count = 0;
    //only S_T and drift vars are allowed to be negative
    for(unsigned int j = 0 ; j < STATE_SIZE-MEAS_SIZE; j++) {
        if(indexof(S_T, count) == j) {
            count++;
        } else if(checkme[j] < 0) {
            weightout = 0.0;
            checkme = defaultvector;
            return true;
        }
    }
    return false;
}

aux::vector BoldModel::defmu(unsigned int simul)
{
    aux::vector ret(simul+simul*LVAR_SIZE+GVAR_SIZE);
    for(unsigned int ii = 0 ; ii < simul; ii++) {
        ret[indexof(TAU_S, ii)] = 4.98;
        ret[indexof(TAU_F, ii)] = 8.31;
        ret[indexof(EPSILON, ii)] = 0.069;
        ret[indexof(TAU_0, ii)] = 8.38;
        ret[indexof(ALPHA, ii)] = .189;
        ret[indexof(E_0, ii)] = .635;
        ret[indexof(V_0, ii)] = 1.49e-2;

        ret[indexof(V_T,ii)] = 1;
        ret[indexof(Q_T,ii)] = 1;
        ret[indexof(S_T,ii)] = 0;
        ret[indexof(F_T,ii)]= 1;
    }
    
    for(unsigned int i = 0; i < simul; i++)
        ret[GVAR_SIZE+simul*LVAR_SIZE+i] = 0;
    return ret;
}

aux::vector BoldModel::defvar(unsigned int simul)
{
    aux::vector ret(simul+simul*LVAR_SIZE+GVAR_SIZE, 0);
    
    for(unsigned int ii = 0 ; ii < simul ; ii++) {
        //set the variances for all the variables to 3*sigma
        ret(indexof(TAU_S  ,ii)) = 1.07*1.07;
        ret(indexof(TAU_F  ,ii)) = 1.51*1.51;
        ret(indexof(EPSILON,ii)) = 0.014*.014;
        ret(indexof(TAU_0  ,ii)) = 1.5*1.5;
        ret(indexof(ALPHA  ,ii)) = .004*.004;
        ret(indexof(E_0    ,ii)) = .072*.072;
        ret(indexof(V_0    ,ii)) = .6e-2*.6e-2;

        //Assume they start at 0
        ret(indexof(V_T,ii)) = 1e-20;
        ret(indexof(Q_T,ii)) = 1e-20;
        ret(indexof(S_T,ii)) = 1e-20;
        ret(indexof(F_T,ii)) = 1e-20;
    }
    for(unsigned int i = 0 ; i < simul; i++) {
        ret(GVAR_SIZE+simul*LVAR_SIZE+i) = 0;
    }

    return ret;
}

aux::symmetric_matrix BoldModel::defcov(unsigned int simul)
{
    aux::symmetric_matrix ret(simul+simul*LVAR_SIZE+GVAR_SIZE, 0);
    
    for(unsigned int ii = 0 ; ii < simul ; ii++) {
        //set the variances for all the variables to 3*sigma
        ret(indexof(TAU_S  ,ii), indexof(TAU_S  ,ii)) = 1.07*1.07;
        ret(indexof(TAU_F  ,ii), indexof(TAU_F  ,ii)) = 1.51*1.51;
        ret(indexof(EPSILON,ii), indexof(EPSILON,ii)) = 0.014*.014;
        ret(indexof(TAU_0  ,ii), indexof(TAU_0  ,ii)) = 1.5*1.5;
        ret(indexof(ALPHA  ,ii), indexof(ALPHA  ,ii)) = .004*.004;
        ret(indexof(E_0    ,ii), indexof(E_0    ,ii)) = .072*.072;
        ret(indexof(V_0    ,ii), indexof(V_0    ,ii)) = .6e-2*.6e-2;

        //Assume they start at 0
        ret(indexof(V_T,ii), indexof(V_T,ii)) = 1e-20;
        ret(indexof(Q_T,ii), indexof(Q_T,ii)) = 1e-20;
        ret(indexof(S_T,ii), indexof(S_T,ii)) = 1e-20;
        ret(indexof(F_T,ii), indexof(F_T,ii)) = 1e-20;
    }
    for(unsigned int i = 0 ; i < simul; i++) {
        ret(GVAR_SIZE+simul*LVAR_SIZE+i, GVAR_SIZE+simul*LVAR_SIZE+i) = 0;
    }

    return ret;
}
