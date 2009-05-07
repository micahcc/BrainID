#ifndef BOLDMODEL_H
#define BOLDMODEL_H

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/aux/DiracMixturePdf.hpp>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <iomanip>

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

namespace aux = indii::ml::aux;

void outputVector(std::ostream& out, aux::vector vec);
void outputMatrix(std::ostream& out, aux::matrix mat);

class BoldModel : public indii::ml::filter::ParticleFilterModel<double>
{
public:
    ~BoldModel();
    BoldModel(aux::vector u = aux::zero_vector(INPUT_SIZE));

    unsigned int getStateSize() { return SYSTEM_SIZE; };
    unsigned int getStimSize() { return INPUT_SIZE; };
    unsigned int getMeasurementSize() { return MEAS_SIZE; };

    int transition(aux::vector& s,
            const double t, const double delta);
    int transition(aux::vector& s,
            const double t, const double delta, const aux::vector& u);

    aux::vector measure(const aux::vector& s);

    //acquire the weight of the particle based on the input
    double weight(const aux::vector& s, const aux::vector& y);

    void generatePrior(aux::DiracMixturePdf&, int);

    //since the particle filter doesn't yet support input, we are
    //going to hack around that and set it directly
    void setinput(aux::vector& in) { input = in; };
    
    //Constants
    static const int THETA_SIZE = 7;
    static const int STATE_SIZE = 4;
    static const int SIMUL_STATES = 1;
    static const int SYSTEM_SIZE = 11;

    static const int MEAS_SIZE = 1;
    static const int INPUT_SIZE = 1;
    static const int STEPS = 250;
    
private:
    //the standard deviations for the parameters theta, which are
    //theoretically constant for the whole volume
//    aux::vector theta_sigmas;
    
    //storage for the current input
    aux::vector input;

    //goes to the index of the given state
    inline int indexof(int name, int index){
        return THETA_SIZE + index*STATE_SIZE + name;
    };

    void generate_component(gsl_rng* rng, aux::vector& fillme);

    //Internal Constants
    static const double A1 = 3.4;
    static const double A2 = 1.0;
    //these are at the beginning of the state array so this assigns the indices
    enum Theta { TAU_S=0, TAU_F=1, EPSILON=2, TAU_0=3 , ALPHA=4, E_0=5, V_0=6};

    //these are recurring in the state array, so V_T could be at 0,4,...16,20...
    enum StateVar { V_T=0, Q_T=1, S_T=2, F_T =3 };

    //variance to apply to 
    double var_e;
    double sigma_e;
//    double small_g;
};

#endif
