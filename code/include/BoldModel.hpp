#ifndef BOLDMODEL_H
#define BOLDMODEL_H

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/aux/DiracMixturePdf.hpp>
#include <gsl/gsl_randist.h>

namespace aux = indii::ml::aux;

class BoldModel : public indii::ml::filter::ParticleFilterModel<double>
{
public:
    ~BoldModel();
    BoldModel(aux::vector stddev, bool expweight, 
                size_t sections = 1, aux::vector drift = aux::vector(1, 0));

    unsigned int getStateSize() const { return STATE_SIZE; };
    unsigned int getStimSize() const { return INPUT_SIZE; };
    unsigned int getMeasurementSize() const { return MEAS_SIZE; };

    int transition(aux::vector& s,
            const double t, const double delta) const;
    int transition(aux::vector& s,
            const double t, const double delta, const aux::vector& u) const;

    aux::vector measure(const aux::vector& s) const;

    //acquire the weight of the particle based on the input
    double weight(const aux::vector& s, const aux::vector& y) const;

    void generatePrior(aux::DiracMixturePdf&, int, double scale = 1) const;
    void generatePrior(aux::DiracMixturePdf& x0, int samples, const aux::vector mean,
          double scale = 1) const;
    void generatePrior(aux::DiracMixturePdf& x0, int samples, 
                const aux::symmetric_matrix cov) const;
    void generatePrior(aux::DiracMixturePdf& x0, int samples, const aux::vector mean,
                const aux::symmetric_matrix cov) const;

    //since the particle filter doesn't yet support input, we are
    //going to hack around that and set it directly
    void setinput(aux::vector in) { input = in; };
    
    bool reweight(aux::vector& checkme, double& weight) const;
    
    //these are recurring in the state array, so V_T could be at 0,4,...16,20...
    //also note that the last MEAS_SIZE members of the state variable are
    //related to drift compensation
    enum StateName { TAU_0=0 , ALPHA=1, E_0=2, V_0=3, TAU_S=4, TAU_F=5, 
                EPSILON=6, V_T=7, Q_T=8, S_T=9, F_T =10};

    const aux::vector& getdefault() const { return defaultstate; };
    void setdefault(aux::vector def) { defaultstate = def; };

    aux::vector estMeasVar(aux::DiracMixturePdf& in) const;
    aux::vector estMeasMean(aux::DiracMixturePdf& in) const;

    //goes to the index of the given state
    static inline size_t indexof(int name, int index) {
        return (name < (int)GVAR_SIZE) ? name : index*LVAR_SIZE + name;
    };

    static double getA1() {return A1;};
    static double getA2() {return A2;};

    static aux::vector defmu(unsigned int);
    static aux::vector defvar(unsigned int);
    static aux::symmetric_matrix defcov(unsigned int);

private:
    //the standard deviations for the parameters theta, which are
    //theoretically constant for the whole volume
//    aux::vector theta_sigmas;
    
    /* Default values for state, known to be stable */
    aux::vector defaultstate;

    //storage for the current input
    aux::vector input;

    void generateComponent(gsl_rng* rng, aux::vector& fillme, 
                const double k_sigma[], const double theta_mu[]) const;

    //Weighting
    enum WeightF { NORM = 0, EXP = 1, HYP = 2} ;
    int weightf;

    //variance to apply to 
    aux::vector sigma;
    
    //Constants
    static const unsigned int GVAR_SIZE = 4;
    static const unsigned int LVAR_SIZE = 7;
    const unsigned int SIMUL_STATES;
    const unsigned int STATE_SIZE;

    const unsigned int MEAS_SIZE;
    const unsigned int INPUT_SIZE;
    
    //Internal Constants
    static const double A1 = 3.4;
    static const double A2 = 1.0;
};

#endif
