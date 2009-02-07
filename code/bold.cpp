#include <cstdio>
#include <itkMatrix.h>
#include "bold.h"
#include "particle.h"

itk::Matrix< double, 4, 10 > weights_g;

bold::bold()
{
    curr_params.Fill(0);
    curr_state.Fill(0);
}

State_t bold::step(State_t prev)
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
    State_t result;

    //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
    result[0] = (1./tau_0) * (f_t - pow(v_t, 1./alpha));
    
    //Q_t* = ...
    double A = (f_t / e_0) * (1 - pow( 1. - e_0, 1./f_t));
                (1 - pow(1-curr_params[5], 1./f_t));
    double B = q_t / pow(v_t, 1.-1./alpha);
    result[1] = (1./tau_0) * ( A - B );

    //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
    result[2] = epsilon*stimuli - s_t/tau_s - (f_t - 1.) / tau_f;

    result[3] = s_t;

    //X_t = X_{t-1} + dx*dt
    result = prev + result*delta_t;
    return result;
}

double error(State_t prev)
{
    
    return 0.0;
}
