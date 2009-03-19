#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"

#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

#define SECTION_SIZE 1
#define TIME_SIZE 100

struct parameters
{
    double tau_s;
    double tau_f;
    double epsilon;
    double tau_0;
    double alpha;
    double e_0;
    double v_0;
}

int func (double t, const double y[], double f[], void *params)
{
    parameters* theta = (parameters*)params;
    f[0] = y[1];
    f[1] = -y[0] - mu*y[1];
    f[2] = -y[0] - mu*y[1];
    f[3] = -y[0] - mu*y[1];
    return GSL_SUCCESS;
}

int jac (double t, const double y[], double *dfdy, double dfdt[], void *params)
{
    parameters* theta = (parameters *)params;
    gsl_matrix_view dfdy_mat 
        = gsl_matrix_view_array (dfdy, 2, 2);
    gsl_matrix * m = &dfdy_mat.matrix; 
    gsl_matrix_set (m, 0, 0, 0.0);
    gsl_matrix_set (m, 0, 1, 1.0);
    gsl_matrix_set (m, 1, 0, -2.0*mu*y[0]*y[1] - 1.0);
    gsl_matrix_set (m, 1, 1, -mu*(y[0]*y[0] - 1.0));
    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    return GSL_SUCCESS;
}

   
int main (void)
{
    //create a 2D output image of appropriate size.
    itk::ImageFileWriter< Image2DType >::Pointer writer = 
        itk::ImageFileWriter< Image2DType >::New();
    Image2DType::Pointer outputImage = Image2DType::New();

    Image2DType::RegionType out_region;
    Image2DType::IndexType out_index;
    Image2DType::SizeType out_size;
    
    out_size[0] = SECTION_SIZE;
    out_size[1] = TIME_SIZE + 1;
    
    out_index[0] = 0;
    out_index[1] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    
    //setup iterator
    itk::ImageLinearIteratorWithIndex<Image2DType> 
                out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetDirection(0);

    const gsl_odeiv_step_type * T 
        = gsl_odeiv_step_rk8pd;

    gsl_odeiv_step * s 
        = gsl_odeiv_step_alloc (T, 2);
    gsl_odeiv_control * c 
        = gsl_odeiv_control_y_new (1e-6, 0.0);
    gsl_odeiv_evolve * e 
        = gsl_odeiv_evolve_alloc (2);

    struct parameters theta;
    theta.tau_s;
    theta.tau_f;
    theta.epsilon;
    theta.tau_0;
    theta.alpha;
    theta.e_0;
    theta.v_0;
    gsl_odeiv_system sys = {func, jac, 4, &theta};

    double t = 0.0, t1 = 100.0;
    double h = 1e-6;
    double y[2] = { 1.0, 0.0 };

    while (t < t1)
    {
        int st
            break;

        printf ("%.5e %.5e %.5e\n", t, y[0], y[1]);
    }



    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);


    return 0;
}

