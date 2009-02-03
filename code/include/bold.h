#ifndef BOLD_H
#define BOLD_H

#include <itkMatrix.h>
#include <itkVector.h>

#include <itkOrientedImage.h>

#include <cmath>

#define NUM_WIENER 10;

//class pair
//{
//public:
//    std::string name;
//    double value;
//}

///////////////////////
//Parameter locations:
//0: \epsilon
//1: \tau_s
//2: \tau_f
//3: \tau_0
//4: \alpha
//5: E_0
//6: V_0
typedef itk::Vector<double, 7> Param_t;

///////////////////////
//State Locations:
//0: v - blood volume
//1: q - normalized deoxyhaemoglobin content
//2: s - flow inducing signal
//3: f - normalized cerebral blood flow
typedef itk::Vector<double, 4> State_t;
typedef itk::Vector<double, 4> Grad_t;


typedef itk::OrientedImage<double, 3> Fmri_t;

//extern itk::Matrix<double, 4, NUM_WIENER> weights_g;

class bold
{
public:
    bold();
    int calc_gradient(State_t curr_state, Fmri_t curr_fmri, Grad_t result);


}
#endif

