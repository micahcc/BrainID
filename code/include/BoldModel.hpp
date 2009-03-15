#ifndef BOLDMODEL_H
#define BOLDMODEL_H

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <iostream>
#include <iomanip>

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

namespace aux = indii::ml::aux;

void outputVector(std::ostream& out, aux::vector vec);
void outputMatrix(std::ostream& out, aux::matrix mat);

class BoldModel : public indii::ml::filter::ParticleFilterModel<double>
{
public:
    ~BoldModel();
    BoldModel();

    unsigned int getStateSize() { return SYSTEM_SIZE; };
    unsigned int getStimSize() { return INPUT_SIZE; };
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

#endif
