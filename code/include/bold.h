#ifndef BOLD_H
#define BOLD_H

#include <itkMatrix.h>
#include <itkVector.h>

#include <itkOrientedImage.h>

#include <cmath>
#include "smctc.hh"

#define NUM_WIENER 10;

//This is a type for holding input from the file or whatever means
//each time point is taken.
struct Input_t
{
    typedef struct {
        unsigned int id;
        double value;
    } data_t;

    unsigned int num_sect;
    double timestep;

    data_t* data;
};

//epsilon  - array[0]
//tau_s    - array[1]
//tau_f    - array[2]
//tau_0    - array[3]
//alpha    - array[4]
//e_0      - array[5]
//v_0      - array[6]
union Param_t
{
    struct {
        double epsilon;
        double tau_s;
        double tau_f;
        double tau_0;
        double alpha;
        double e_0;
        double v_0;
    } named;

    double array[7];
};

//v_t  -  array[0]
//q_t  -  array[1]
//s_t  -  array[2]
//f_t  -  array[3]
union State_t
{
    struct
    {
        double v_t;
        double q_t;
        double s_t;
        double f_t;

    } named;

    double array[4];
};

struct ExtraData_t
{
    Param_t params;
    double control;
    double delta_t;
    double observed;
    double var_e;
    double gg;
    double a1;
    double a2;
};

int read_header(FILE* file, Input_t* input);

int read_time(FILE* file, Input_t* input);

//step provides the pr(X_t|X_t-1)
//state is a state variable, which is a 1 dimensional vector of constant
//      length.
//extras is just a data structure that was passed in
//dist is a a set of tuples (len 2) where dist[i][0] is the mean and dist[i][1]
//      is the variance. Obviously every member of state will have a different
//      mean and variance. This is the output of the function
void step(const double* state_a, void* extras, double dist[][2]);

//error provides the pr(y_t | x_t-1)
void error(const double* state_a, double dist[2], void* extras);

smc::particle<State_t> fInitialize(smc::rng *pRng);
void fMove(long lTime, smc::particle<State_t> & pFrom, smc::rng *pRng);

#endif

