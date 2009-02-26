#include <cstdio>
#include <itkMatrix.h>
#include "bold.h"
#include "particle.h"

int read_header(FILE* file, Input_t* input)
{
    //read the first couple lines to get the number of sections
    //and the time step size
    return 0;
}

int read_time(FILE* file, Input_t* input)
{
    //read the necessary bytes to get all the data
    //for a time step
    return 0;
}

//step provides the pr(X_t|X_t-1)
//state is a state variable, which is a 1 dimensional vector of constant
//      length.
//extras is just a data structure that was passed in
//dist is a a set of tuples (len 2) where dist[i][0] is the mean and dist[i][1]
//      is the variance. Obviously every member of state will have a different
//      mean and variance. This is the output of the function
void step(const double* state_a, void* extras, double dist[][2])
{
    State_t tmp;
    State_t* state = (State_t*) state_a;

    ExtraData_t* extradata = (ExtraData_t*) extras;
    Param_t* params = &extradata->params;

    //V_t* = (1/tau_0) * ( f_t - v_t ^ (1/\alpha)) 
    tmp.named.v_t = (1./params->named.tau_0) * (state->named.f_t -
                pow(state->named.v_t, 1./params->named.alpha));
    
    //Q_t* = ...
    double A = (state->named.f_t / params->named.e_0) * 
                (1 - pow( 1. - params->named.e_0, 1./state->named.f_t));
    double B = state->named.q_t / pow(state->named.v_t, 1.-1./params->named.alpha);
    tmp.named.q_t = (1./params->named.tau_0) * ( A - B );

    //S_t* = \epsilon*u_t - 1/\tau_s * s_t - 1/\tau_f * (f_t - 1)
    tmp.named.s_t = params->named.epsilon*extradata->control - 
                state->named.s_t/params->named.tau_s - 
                (state->named.f_t - 1.) / params->named.tau_f;

    tmp.named.f_t = state->named.s_t;

    //X_t = X_{t-1} + dx*dt
    for(int i = 0 ; i < 4 ; i++) {
        dist[i][0] = state->array[i] + tmp.array[i]*extradata->delta_t;
        dist[i][1] = extradata->gg*extradata->delta_t;
    }
}

//error provides the pr(y_t | x_t-1)
void error(const double* state_a, double dist[2], void* extras)
{
    ExtraData_t* extradata = (ExtraData_t*) extras;
    Param_t* params = &extradata->params;
    State_t* state = (State_t*) state_a;
    
    double expected = params->named.v_0 * 
                (extradata->a1*(1-state->named.q_t) -
                extradata->a2*(1-state->named.v_t));
    double diff = extradata->observed - expected;
    dist[0] = diff > 0 ? diff : -diff;
    dist[1] = extradata->var_e;
}

smc::particle<State_t> fInitialize(smc::rng *pRng)
{
    

}
