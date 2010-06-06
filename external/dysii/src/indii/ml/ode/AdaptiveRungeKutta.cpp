//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "AdaptiveRungeKutta.hpp"

#include <assert.h>
#include <math.h>

#include <gsl/gsl_errno.h>

using namespace indii::ml::ode;

namespace aux = indii::ml::aux;

const gsl_odeiv_step_type* AdaptiveRungeKutta::gslStepType
    = gsl_odeiv_step_rkf45;  // explicit method
  //= gsl_odeiv_step_rk4imp; // implicit method

AdaptiveRungeKutta::AdaptiveRungeKutta(DifferentialModel* model) :
  NumericalSolver(model->getDimensions()), model(model) {
  setStepType(gslStepType);
}

AdaptiveRungeKutta::AdaptiveRungeKutta(DifferentialModel* model,
    const aux::vector& y0) : NumericalSolver(y0), model(model) {
  /* pre-condition */
  assert (y0.size() == model->getDimensions());

  setStepType(gslStepType);
}

AdaptiveRungeKutta::~AdaptiveRungeKutta() {
  //
}

int AdaptiveRungeKutta::calculateDerivativesForward(double t,
    const double y[], double dydt[]) {
  model->calculateDerivatives(t, y, dydt);

  return GSL_SUCCESS;
}

int AdaptiveRungeKutta::calculateDerivativesBackward(double t,
    const double y[], double dydt[]) {
  /* convert the proposed step to a future time into a proposed step
     to a past time */
  int result = calculateDerivativesForward(2.0 * base - t, y, dydt);

  if (result == GSL_SUCCESS) {
    /* as we're moving backward in time, negate gradients */
    size_t i;
    for (i = 0; i < dimensions; i++) {
      dydt[i] *= -1.0;
    }
  }

  return result;
}

