#ifndef BOLDMODEL_H
#define BOLDMODEL_H

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/aux/DiracMixturePdf.hpp>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <iomanip>
#include <vector>

//State Consists of Two or More Sections:
//Theta
//4 - tau_s
//5 - tau_f
//8 - epsilon
//3 - tau_0
//6 - alpha
//7 - E_0
//0 - V_0
//Actual States
//9+4*i+0 - v_t
//9+4*i+1 - q_t
//9+4*i+2 - s_t
//9+4*i+3 - f_t

typedef struct 
{
    unsigned int index; //start index in state variable
    std::vector< unsigned int > slice_index; //start index of slice variables
} SectionDesc;

namespace aux = indii::ml::aux;

class BoldModel : public indii::ml::filter::ParticleFilterModel<double>
{
public:
    ~BoldModel();
    BoldModel(aux::vector stddev, bool weightf = false,
                size_t sections = 1,
                aux::vector u = aux::zero_vector(1));

    virtual unsigned int getStateSize() { return STATE_SIZE; };
    unsigned int getStimSize() { return INPUT_SIZE; };
    virtual unsigned int getMeasurementSize() { return MEAS_SIZE; };

    int transition(aux::vector& s,
            const double t, const double delta);
    int transition(aux::vector& s,
            const double t, const double delta, const aux::vector& u);

    aux::vector measure(const aux::vector& s);

    //acquire the weight of the particle based on the input
    double weight(const aux::vector& s, const aux::vector& y);

    void generatePrior(aux::DiracMixturePdf&, int, double scale = 1);
    void generatePrior(aux::DiracMixturePdf& x0, int samples, const aux::vector mean,
          double scale = 1);
    void generatePrior(aux::DiracMixturePdf& x0, int samples, 
                const aux::symmetric_matrix cov);
    void generatePrior(aux::DiracMixturePdf& x0, int samples, const aux::vector mean,
                const aux::symmetric_matrix cov);

    //since the particle filter doesn't yet support input, we are
    //going to hack around that and set it directly
    void setinput(aux::vector& in) { input = in; };
    
    bool reweight(aux::vector& checkme, double& weight);
    
    //these are recurring in the state array, so V_T could be at 0,4,...16,20...
    enum StateName { TAU_0=0 , ALPHA=1, E_0=2, V_0=3, TAU_S=4, TAU_F=5, 
                EPSILON=6, V_T=7, Q_T=8, S_T=9, F_T =10};

    const aux::vector& getdefault() { return defaultstate; };
    void setdefault(aux::vector def) { defaultstate = def; };

    aux::vector estMeasVar(aux::DiracMixturePdf& in);
    aux::vector estMeasMean(aux::DiracMixturePdf& in);

    //goes to the index of the given state
    inline size_t indexof(int name, int index){
        return (name < (int)GVAR_SIZE) ? name : index*LVAR_SIZE + name;
    };

private:
    //the standard deviations for the parameters theta, which are
    //theoretically constant for the whole volume
//    aux::vector theta_sigmas;
    
    /* Default values for state, known to be stable */
    aux::vector defaultstate;

    //storage for the current input
    aux::vector input;

    void generate_component(gsl_rng* rng, aux::vector& fillme, 
                const double k_sigma[], const double theta_mu[]);

    //Weighting
    enum WeightF { NORM = 0, EXP = 1, HYP = 2} ;
    int weightf;

    //variance to apply to 
    aux::vector sigma;
    
    //Constants
    const unsigned int GVAR_SIZE;
    const unsigned int LVAR_SIZE;
    const unsigned int SIMUL_STATES;
    const unsigned int STATE_SIZE;

    const unsigned int MEAS_SIZE;
    const unsigned int INPUT_SIZE;
    
    //Internal Constants
    static const double A1 = 3.4;
    static const double A2 = 1.0;
};

#endif
