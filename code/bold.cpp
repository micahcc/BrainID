#include <cstdio>
#include <itkMatrix.h>
#include "bold.h"
#include "particle.h"

itk::Matrix< double, 4, 10 > weights_g;

//Current state represents the vector X, gradient X*, params theta,
//and stimuli, u, where X_t* = X_{t-1} + f(X_t, u_t, theta)*dt
int calc_gradient(State_t curr_state, double stimuli, Param_t curr_params, 
            Grad_t* result)
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
    result[0] = (1./tau_0) * (f_t - pow(v_t, 1./alpha));
    
    //Q_t* = ...
    double A = (f_t / e_0) * (1 - pow( 1. - e_0, 1./f_t));
                (1 - pow(1-curr_params[5], 1./curr_state[3]));
    double B = q_t / pow(v_t, 1.-1./alpha);
    result[1] = (1./tau_0) * ( A - B );

    //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
    result[2] = epsilon*stimuli - s_t/tau_s - (f_t - 1.) / tau_f;

    result[3] = s_t;

    return 0;
}

