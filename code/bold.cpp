#include <cstdio>
#include <itkMatrix.h>
#include "bold.h"
#include "particle.h"

itk::Matrix< double, 4, 10 > weights_g;

int calc_gradient(State_t curr_state, Fmri_t curr_fmri, Param_t curr_params, 
            Grad_t result)
{
    #define epsilon  curr_params[0]
    #define tau_s    curr_params[1]
    #define tau_f    curr_params[2]
    #define tau_0    curr_params[3]
    #define alpha    curr_params[4]
    #define e_0      curr_params[5]
    #define v_0      curr_params[6]

    #define v_t      curr_state[0]
    #define q_t      curr_state[1]
    #define s_t      curr_state[2]
    #define f_t      curr_state[3]

    //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
    result[0] = (1./tau_0) * (curr_state[3] - pow(curr_state[0], 1./curr_params[4]));
    
    //Q_t* = ...
    double A = (curr_state[3]/curr_params[5]) * (1 - pow(1-curr_params[5], 1./curr_state[3]));
    double B = curr_state[1]/pow(curr_state[0], 1.-1./curr_params[4]);
    result[1] = (1./curr_params[3]) * ( A - B );
    return 0;
}

