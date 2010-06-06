#ifndef INDII_ML_SDE_STOCHASTICADAPTIVEEULERMARUYAMA_HPP
#define INDII_ML_SDE_STOCHASTICADAPTIVEEULERMARUYAMA_HPP

#include "StochasticDifferentialModel.hpp"
#include "StochasticNumericalSolver.hpp"

namespace indii {
  namespace ml {
    namespace sde {    
/**
 * Stochastic Adaptive Euler-Maruyama method for solving a system of
 * stochastic differential equations.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 582 $
 * @date $Date: 2008-12-15 17:03:50 +0000 (Mon, 15 Dec 2008) $
 *
 * @param DT Type of the diffusion matrix.
 * @param DDT Type of the diffusion partial derivative matrices.
 *
 * This class numerically solves StochasticDifferentialModel models
 * defining a system of stochastic differential equations using an
 * Euler-Maruyama scheme. Error is approximated by comparison with two half
 * steps, and step size doubled or halved accordingly.
 *
 * The general usage idiom is as for
 * indii::ml::ode::AdaptiveRungeKutta. The class will automatically
 * keep track of Wiener process increments.
 *
 * @todo This class is tightly coupled with the GSL and would benefit from
 * greater independence.
 */
template <class DT = indii::ml::aux::matrix,
    class DDT = indii::ml::aux::zero_matrix>
class StochasticAdaptiveEulerMaruyama :
    public indii::ml::sde::StochasticNumericalSolver {
public:
  /**
   * Constructor.
   *
   * @param model Model to estimate.
   *
   * The time is initialised to zero, but the state is uninitialised
   * and should be set with setVariable() or setState().
   */
  StochasticAdaptiveEulerMaruyama(StochasticDifferentialModel<DT,DDT>* model);

  /**
   * Constructor.
   *
   * @param model Model to estimate.
   * @param y0 Initial state.
   *
   * The time is initialised to zero and the state to that given.
   */
  StochasticAdaptiveEulerMaruyama(StochasticDifferentialModel<DT,DDT>* model,
      const indii::ml::aux::vector& y0);

  /**
   * Destructor.
   */
  virtual ~StochasticAdaptiveEulerMaruyama();

  virtual double step(double upper);

  virtual double stepBack(double lower);

  virtual int calculateDerivativesForward(double t, const double y[],
      double dydt[]);

  virtual int calculateDerivativesBackward(double t, const double y[],
      double dydt[]);

private:
  /**
   * Model.
   */
  StochasticDifferentialModel<DT,DDT>* model;

  /**
   * Drift.
   */
  indii::ml::aux::vector a1;
  
  /**
   * Drift at midpoint.
   */
  indii::ml::aux::vector a_mid;
  
  /**
   * Diffusion.
   */
  DT B1;
  
  /**
   * Diffusion at midpoint.
   */
  DT B_mid;

};

    }
  }
}

#include "boost/numeric/ublas/operation.hpp"
#include "boost/numeric/ublas/operation_sparse.hpp"

template <class DT, class DDT>
indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::StochasticAdaptiveEulerMaruyama(
    StochasticDifferentialModel<DT,DDT>* model) :
    StochasticNumericalSolver(model->getDimensions(),
    model->getNoiseDimensions()), model(model),
    a1(model->getDimensions()), a_mid(model->getDimensions()),
    B1(model->getDimensions(), model->getNoiseDimensions()),
    B_mid(model->getDimensions(), model->getNoiseDimensions()) {
  setStepType(gsl_odeiv_step_gear1);
}

template <class DT, class DDT>
indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::StochasticAdaptiveEulerMaruyama(
    StochasticDifferentialModel<DT,DDT>* model,
    const indii::ml::aux::vector& y0) : StochasticNumericalSolver(y0,
    model->getNoiseDimensions()), model(model),
    a1(model->getDimensions()), a_mid(model->getDimensions()),
    B1(model->getDimensions(), model->getNoiseDimensions()),
    B_mid(model->getDimensions(), model->getNoiseDimensions()) {

  /* pre-condition */
  assert (y0.size() == model->getDimensions());

  setStepType(gsl_odeiv_step_gear1);
}

template <class DT, class DDT>
indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::~StochasticAdaptiveEulerMaruyama() {
  //
}

template <class DT, class DDT>
double indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::step(
    double upper) {
  /* pre-condition */
  assert (upper > t);

  const unsigned int N = model->getDimensions();
  const unsigned int V = model->getNoiseDimensions();
  double ts, ts_new, ts_mid, ts_prev = 0.0;
  double h, h_new, h_mid;
  double absErr, relErr;
  bool stepDecreased, numSound;
  
  /* see ode_initval/cstd.c for how state struct is laid out */
  double* state = (double*)gslControl->state;
  absErr = state[0];
  relErr = state[1];

  /* make sure step size won't exceed upper bound */
  if (maxStepSize > 0.0 && stepSize > maxStepSize) {
    stepSize = maxStepSize;
  }
  ts = t + stepSize;
  if (ts > upper) {
    ts = upper;
  }

  /* stepping variables */
  aux::vector y(this->getState());
  aux::vector y1(N), y_mid(N), y2(N);
  aux::vector dW1(V), dW_mid(V);
  aux::vector epsilon(N), delta(N);
  unsigned int i;

  /* new sample of Wiener process */
  sampleNoise(&ts);
  h = ts - t;

  /* full step */
  noalias(dW1) = dWf.top();  
  model->calculateDrift(t, y, a1);
  model->calculateDiffusion(t, y, B1);
  noalias(y1) = y + h*a1;
  ublas::axpy_prod(B1, dW1, y1, false);

  ts_mid = 0.5*(ts + t);
    
  stepDecreased = true; // for loop initialisation
  numSound = sampleNoise(&ts_mid);
  while (stepDecreased && numSound) {
    h_mid = ts_mid - t;

    noalias(dW_mid) = dWf.top();
    noalias(y_mid) = y + h_mid*a1;
    ublas::axpy_prod(B1, dW_mid, y_mid, false);

    model->calculateDrift(t + h_mid, y_mid, a_mid);
    model->calculateDiffusion(t + h_mid, y_mid, B_mid);
    noalias(y2) = y_mid + h_mid*a_mid;
    ublas::axpy_prod(B_mid, dW1-dW_mid, y2, false);

    /* error control */
    /* normal case */
    for (i = 0; i < N; i++) {
      epsilon(i) = fabs(y1(i) - y2(i));
      delta(i) = absErr + relErr*fabs(y1(i));
    }
    stepDecreased = ublas::norm_inf(element_div(epsilon, delta)) > 1.0;
    
    if (stepDecreased) {
      ts_prev = ts;
      ts = ts_mid;
      h = h_mid;
      noalias(dW1) = dW_mid;
      noalias(y1) = y_mid;
      
      ts_mid = 0.5*(ts + t);
      numSound = sampleNoise(&ts_mid);
    }
  }

  /* update proposed step size */
  if (ts_prev > 0.0) {
    stepSize = ts_prev - t; // better numerically
  } else {
    stepSize = 2.0*(ts - t);
  }

  /* update state */
  aux::vectorToArray(y1, this->y); // don't use setState(), clears tf and dWf

  /* update time */
  t = ts;

  /* clean up future path */
  if (numSound) {
    tf.pop(); // half step sample
    dWf.pop();
  }
  
  tf.pop(); // full step sample
  dWf.pop();

  /* post-condition */
  assert (tf.empty() || t < tf.top());
  assert (t <= upper);

  return t;
}

template <class DT, class DDT>
inline double indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::stepBack(
    double lower) {
  double t = NumericalSolver::stepBack(lower);

  return t;
}

template <class DT, class DDT>
inline int indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::calculateDerivativesForward(
    double ts, const double y[], double dydt[]) {
  /* not required by this implementation */
  assert (false);
  
  return GSL_FAILURE;
}

template <class DT, class DDT>
inline int indii::ml::sde::StochasticAdaptiveEulerMaruyama<DT,DDT>::calculateDerivativesBackward(
    double t, const double y[], double dydt[]) {
  /* not required by this implementation */
  assert (false);
  
  return GSL_FAILURE;
}

#endif

