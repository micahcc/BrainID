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
    BoldModel(aux::vector stddev, int weightfunc, size_t sections = 1);

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

    void generatePrior(aux::DiracMixturePdf&, int, double scale = 1, 
                bool flat = true) const;
    void generatePrior(aux::DiracMixturePdf& x0, int samples, const aux::vector mean,
                double scale = 1, bool flat = true) const;
    void generatePrior(aux::DiracMixturePdf& x0, int samples, 
                aux::vector width, bool flat = true) const;
    void generatePrior(aux::DiracMixturePdf& x0, int samples, const aux::vector mean,
                aux::vector width, bool flat = true) const;

    //since the particle filter doesn't yet support input, we are
    //going to hack around that and set it directly
    void setinput(aux::vector in) { input = in; };
    
    bool reweight(aux::vector& checkme, double& weight) const;
    
    //these are recurring in the state array, so V_T could be at 0,4,...16,20...
    //also note that the last MEAS_SIZE members of the state variable are
    //related to drift compensation
    enum StateName { TAU_0=0 , ALPHA=1, E_0=2, V_0=3, TAU_S=4, TAU_F=5, 
                EPSILON=6, V_T=7, Q_T=8, S_T=9, F_T =10};
    
    enum WeightF { NORM = 0, LAPLACE = 1, HYP = 2, CAUCHY = 3} ;

    static aux::vector getA(double E_0);

    //goes to the index of the given state
    static inline unsigned int indexof(unsigned int name, unsigned int index) {
        return (name < (int)GVAR_SIZE) ? name : index*LVAR_SIZE + name;
    };

    static inline unsigned int stateAt(unsigned int ii) {
        return ii < GVAR_SIZE ? ii : (ii - GVAR_SIZE)%LVAR_SIZE + GVAR_SIZE;
    }

    static aux::vector defmu(unsigned int);
    static aux::vector defsigma(unsigned int);

private:
    //the standard deviations for the parameters theta, which are
    //theoretically constant for the whole volume
//    aux::vector theta_sigmas;
    
    /* Default values for state, known to be stable */
    aux::vector defaultstate;

    //storage for the current input
    aux::vector input;

    double generateComponent(gsl_rng* rng, aux::vector& fillme, 
                aux::vector scale, aux::vector loc) const;

    //Weighting
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
//    const double EPSILON_0 = 1.43;
//    const double NU_0 = 40.3;
//    const double TE = .04; //40ms
    static const double k1 = 4.3*40.3*.04 ;
    static const double k2 = 1.43*25*.04;
    static const double k3 = 1.43-1;
};

#endif
