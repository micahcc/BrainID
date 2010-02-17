#include "BoldModel.hpp"

#include <indii/ml/aux/matrix.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/DiracPdf.hpp>

#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>

BoldModel::BoldModel(aux::vector stddev, bool expweight, 
            size_t sections, aux::vector drift) :
            sigma(stddev), GVAR_SIZE(4), LVAR_SIZE(7), SIMUL_STATES(sections),
            STATE_SIZE(4+7*sections + sections),
            MEAS_SIZE(sections),
            INPUT_SIZE(1)//, segments(sections)
{
    //this is only a problem if the user put in a bad vector
    //in which case the u will be overwritten with 0's
    this->input = aux::zero_vector(sections);
    
    if(expweight) 
        this->weightf = EXP;;

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
        drift = aux::vector(MEAS_SIZE, 1e-10);
    for(unsigned int i = 0; i < MEAS_SIZE ; i++)
        defaultstate[STATE_SIZE-MEAS_SIZE+i] = drift[i];

    std::cerr << "Sizes:" << std::endl;
    std::cerr << this->GVAR_SIZE        << std::endl;
    std::cerr << this->LVAR_SIZE        << std::endl;
    std::cerr << this->SIMUL_STATES     << std::endl;
    std::cerr << this->STATE_SIZE       << std::endl;
    std::cerr << this->MEAS_SIZE        << std::endl;
    std::cerr << this->INPUT_SIZE       << std::endl;

    std::cerr << "Sigma:" << std::endl;
    for(unsigned int i = 0 ; i < sigma.size() ; i++)
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
    if(isnan(tmp-tmp)) {
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

    if(weightf == EXP) {
    	for(unsigned int i = 0 ; i < MEAS_SIZE ; i++)
            weight *= gsl_ran_exponential_pdf(
                        fabs(y[i]-meas[i]+s[STATE_SIZE-MEAS_SIZE+i]),
                        sigma(i));
    } else {
    	for(unsigned int i = 0 ; i < MEAS_SIZE ; i++)
            weight *= gsl_ran_gaussian_pdf(y[i]-meas[i]+s[STATE_SIZE-MEAS_SIZE+i],
                        sigma(i));
    }
    return weight;
}


//Note that k_sigma contains std. deviation OR k and theta_mu contains either
//mean or theta depending on the distribution
void BoldModel::generateComponent(gsl_rng* rng, aux::vector& fillme, 
            const double* k_sigma, const double* theta_mu) const
{
    //going to distribute all the state variables the same even if they are
    //in different sections
    int count = 0;
    for(size_t i = 0 ; i< STATE_SIZE - MEAS_SIZE; i++) {
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
    
    for(size_t i = STATE_SIZE - MEAS_SIZE; i < STATE_SIZE; i++) {
        if(theta_mu[i] != 0) 
            fillme[i] = gsl_ran_gaussian(rng, k_sigma[i]) + theta_mu[i];
    }
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
            double varwidth) const
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

        cov(indexof(V_T,ii), indexof(V_T,ii)) = varwidth*.0001;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = varwidth*.0001;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = varwidth*.0001;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = varwidth*.0001;
    }

    for(unsigned int ii = STATE_SIZE-MEAS_SIZE ; ii < STATE_SIZE; ii++) {
        cov(ii,ii) = mean[ii]*mean[ii]/64;
    }
    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::vector mean, double varwidth) const
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
        cov(indexof(V_T,ii), indexof(V_T,ii)) = varwidth*.0001;
        cov(indexof(Q_T,ii), indexof(Q_T,ii)) = varwidth*.0001;
        cov(indexof(S_T,ii), indexof(S_T,ii)) = varwidth*.0001;
        cov(indexof(F_T,ii), indexof(F_T,ii)) = varwidth*.0001;
    }
    for(unsigned int ii = STATE_SIZE-MEAS_SIZE ; ii < STATE_SIZE; ii++) {
        cov(ii,ii) = mean[ii]*mean[ii]/64;
    }
    
    generatePrior(x0, samples, mean, cov);
}

void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples, 
            const aux::symmetric_matrix cov) const
{
    aux::vector mean = getdefault();
    generatePrior(x0, samples, mean, cov);
}

//TODO make some of these non-gaussian
void BoldModel::generatePrior(aux::DiracMixturePdf& x0, int samples,
            const aux::vector mean, const aux::symmetric_matrix cov) const
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    
    unsigned int count = 0;
    double k_sigma[STATE_SIZE]; 
    double theta_mu[STATE_SIZE];
    for(unsigned int i = 0 ; i < STATE_SIZE-MEAS_SIZE; i++) {
        if(indexof(S_T, count) == i) {
            count++;
            theta_mu[i] = mean(i);
            k_sigma[i] = sqrt(cov(i,i));
        } else {
            theta_mu[i] = cov(i,i)/mean(i);
            k_sigma[i] = mean[i]/theta_mu[i];
        }
    }
    
    for(unsigned int i = STATE_SIZE-MEAS_SIZE; i < STATE_SIZE ; i++) {
        theta_mu[i] = mean(i);
        k_sigma[i] = sqrt(cov(i,i));
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
    aux::vector comp(STATE_SIZE);
    for(int i = 0 ; i < samples; i ++) {
        generateComponent(rng, comp, k_sigma, theta_mu);
        x0.add(comp, 1.0);
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

