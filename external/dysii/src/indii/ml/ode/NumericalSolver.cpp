#include "NumericalSolver.hpp"

#include <assert.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv.h>

using namespace indii::ml::ode;

namespace aux = indii::ml::aux;

NumericalSolver::NumericalSolver(const unsigned int dimensions) :
    dimensions(dimensions) {
  this->t = 0.0;
  this->gslStep = NULL;
  this->gslControl = NULL;
  this->gslEvolve = NULL;
  this->y = new double[dimensions];

  setSuggestedStepSize();
  setMaxStepSize();
  setErrorBounds();
  setDiscontinuity();
  init();
}

NumericalSolver::NumericalSolver(const aux::vector& y0) :
    dimensions(y0.size()) {
  this->t = 0.0;
  this->gslStep = NULL;
  this->gslControl = NULL;
  this->gslEvolve = NULL;
  this->y = new double[dimensions];

  setSuggestedStepSize();
  setMaxStepSize();
  setErrorBounds();
  setDiscontinuity();
  init();
  setState(y0);
}

NumericalSolver::~NumericalSolver() {
  terminate();
  delete[] y;
}

void NumericalSolver::setTime(const double t) {
  this->t = t;
  reset();
}

double NumericalSolver::step(double upper) {
  /* pre-condition */
  assert (upper > t);

  /* solve differential equations */
  int err;
  if (maxStepSize > 0.0 && stepSize > maxStepSize) {
    stepSize = maxStepSize;
  }
  err = gsl_odeiv_evolve_apply(gslEvolve, gslControl, gslStep,
      &gslForwardSystem, &t, upper, &stepSize, y);
  assert (err == GSL_SUCCESS);

  /* post-condition */
  assert (t <= upper);

  return t;
}

void NumericalSolver::stepTo(double to) {
  /* pre-condition */
  assert (t < to);
  
  double p = t;
  while (p < to) {
    p = step(to);
  }
  
  /* post-condition */
  assert (t == to);
}

double NumericalSolver::stepBack(double lower) {
  /* pre-condition */
  assert (lower < t);
  assert (lower >= 0.0);

  /* Solve differential equations, pretending this is a forward step
     and converting the lower bound to an upper bound an equivalent
     distance from the current time. gslBackwardFunction() and
     gslForwardFunction() handle the conversion of the resulting
     forward time step proposals to backward time steps proposals. The
     GSL is only able to make forward steps with its Runge-Kutta
     algorithms. */
  gsl_odeiv_step_reset(gslStep);
  gsl_odeiv_evolve_reset(gslEvolve);

  base = t;
  int err;
  if (stepSize > maxStepSize) {
    stepSize = maxStepSize;
  }
  err = gsl_odeiv_evolve_apply(gslEvolve, gslControl, gslStep,
      &gslBackwardSystem, &t, 2.0 * base - lower, &stepSize, y);
  assert (err == GSL_SUCCESS);
  t = 2.0 * base - t;

  /* post-condition */
  assert (t >= lower);

  return t;
}

void NumericalSolver::stepBackTo(double to) {
  /* pre-condition */
  assert (to < t && to >= 0.0);
  
  double p = t;
  while (p > to) {
    p = stepBack(to);
  }
  
  /* post-condition */
  assert (t == to);
}

void NumericalSolver::setErrorBounds(double maxAbsoluteError,
      double maxRelativeError) {
  if (gslControl == NULL) {
    gslControl = gsl_odeiv_control_standard_new(maxAbsoluteError,
        maxRelativeError, 1.0, 0.0);
  } else {
    gsl_odeiv_control_init(gslControl, maxAbsoluteError, maxRelativeError,
			   1.0, 0.0);
  }
}

void NumericalSolver::setStepSize(double stepSize) {
  this->stepSize = stepSize;
}

void NumericalSolver::setSuggestedStepSize(double stepSize) {
  this->suggestedStepSize = stepSize;
  this->stepSize = stepSize;
}

void NumericalSolver::setMaxStepSize(double stepSize) {
  this->maxStepSize = stepSize;
}

void NumericalSolver::setDiscontinuity() {
  setStepSize(suggestedStepSize);
}

void NumericalSolver::setVariable(const unsigned int index,
      const double value) {
  /* pre-condition */
  assert (index < dimensions);

  y[index] = value;
  reset();
}

void NumericalSolver::setState(const aux::vector& y) {
  /* pre-condition */
  assert (y.size() == dimensions);

  unsigned int i;
  for (i = 0; i < dimensions; i++) {
    this->y[i] = y(i);
  }
  reset();
}

void NumericalSolver::init() {
  /* allocate GSL structures */
  gslEvolve = gsl_odeiv_evolve_alloc(dimensions);

  /* set up systems of differential equations */
  gslForwardSystem.function = gslForwardFunction;
  gslForwardSystem.jacobian = NULL;
  gslForwardSystem.dimension = dimensions;
  gslForwardSystem.params = this;

  gslBackwardSystem.function = gslBackwardFunction;
  gslBackwardSystem.jacobian = NULL;
  gslBackwardSystem.dimension = dimensions;
  gslBackwardSystem.params = this;
}

void NumericalSolver::terminate() {
  /* free GSL structures */
  if (gslStep != NULL) {
    gsl_odeiv_step_free(gslStep);
    gslStep = NULL;
  }
  if (gslControl != NULL) {
    gsl_odeiv_control_free(gslControl);
    gslControl = NULL;
  }
  if (gslEvolve != NULL) {
    gsl_odeiv_evolve_free(gslEvolve);
    gslEvolve = NULL;
  }
}

void NumericalSolver::reset() {
  /* reset GSL structures */
  gsl_odeiv_step_reset(gslStep);
  gsl_odeiv_evolve_reset(gslEvolve);
}

void NumericalSolver::setStepType(const gsl_odeiv_step_type* stepType) {
  if (gslStep != NULL) {
    gsl_odeiv_step_free(gslStep);
  }
  gslStep = gsl_odeiv_step_alloc(stepType, dimensions);
}

int NumericalSolver::gslForwardFunction(double t, const double y[],
    double dydt[], void* params) {
  NumericalSolver* solver = static_cast<NumericalSolver*>(params);
  return solver->calculateDerivativesForward(t, y, dydt);
}

int NumericalSolver::gslBackwardFunction(double t, const double y[],
    double dydt[], void* params) {
  NumericalSolver* solver = static_cast<NumericalSolver*>(params);
  return solver->calculateDerivativesBackward(t, y, dydt);
}

