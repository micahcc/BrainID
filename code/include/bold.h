#ifndef BOLD_H
#define BOLD_H

#include <itkMatrix.h>
#include <itkVector.h>

#include <itkOrientedImage.h>

#include <cmath>

#define NUM_WIENER 10;

//typedef itk::OrientedImage<double, 3> Fmri_t;
//
//class bold
//{
//public:
//    bold();
//    State_t step(State_t);
//    double error(State_t);
//
//private:
//    double delta_t;
//}

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

#endif

