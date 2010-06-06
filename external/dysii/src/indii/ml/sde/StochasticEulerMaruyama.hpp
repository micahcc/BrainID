#ifndef INDII_ML_SDE_STOCHASTICEULERMARUYAMA_HPP
#define INDII_ML_SDE_STOCHASTICEULERMARUYAMA_HPP

#include "StochasticDifferentialModel.hpp"
#include "StochasticNumericalSolver.hpp"

namespace indii {
  namespace ml {
    namespace sde {    
/**
 * Stochastic Euler-Maruyama method with fixed time step for solving a
 * system of stochastic differential equations.
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
 * Euler-Maruyama scheme. The time step is fixed and no error control
 * used.
 *
 * The general usage idiom is as for indii::ml::ode::AdaptiveRungeKutta.
 *
 * @todo This class is tightly coupled with the GSL and would benefit from
 * greater independence.
 */
template <class DT = indii::ml::aux::matrix,
    class DDT = indii::ml::aux::zero_matrix>
class StochasticEulerMaruyama :
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
  StochasticEulerMaruyama(StochasticDifferentialModel<DT,DDT>* model);

  /**
   * Constructor.
   *
   * @param model Model to estimate.
   * @param y0 Initial state.
   *
   * The time is initialised to zero and the state to that given.
   */
  StochasticEulerMaruyama(StochasticDifferentialModel<DT,DDT>* model,
      const indii::ml::aux::vector& y0);

  /**
   * Destructor.
   */
  virtual ~StochasticEulerMaruyama();

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
  indii::ml::aux::vector a;
  
  /**
   * Diffusion.
   */
  DT B;

};

    }
  }
}

#include "boost/numeric/ublas/operation.hpp"
#include "boost/numeric/ublas/operation_sparse.hpp"

template <class DT, class DDT>
indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::StochasticEulerMaruyama(
    StochasticDifferentialModel<DT,DDT>* model) :
    StochasticNumericalSolver(model->getDimensions(),
    model->getNoiseDimensions()), model(model), a(model->getDimensions()),
    B(model->getDimensions(), model->getNoiseDimensions()) {
  a.clear();
  B.clear();
  setStepType(gsl_odeiv_step_gear1);
}

template <class DT, class DDT>
indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::StochasticEulerMaruyama(
    StochasticDifferentialModel<DT,DDT>* model,
    const indii::ml::aux::vector& y0) : StochasticNumericalSolver(y0,
    model->getNoiseDimensions()), model(model), a(model->getDimensions()),
    B(model->getDimensions(), model->getNoiseDimensions()) {
  /* pre-condition */
  assert (y0.size() == model->getDimensions());

  a.clear();
  B.clear();
  setStepType(gsl_odeiv_step_gear1);
}

template <class DT, class DDT>
indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::~StochasticEulerMaruyama() {
  //
}

template <class DT, class DDT>
double indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::step(
    double upper) {
  /* pre-condition */
  assert (upper > t);

  const unsigned int N = model->getDimensions();
  const unsigned int V = model->getNoiseDimensions();
  double ts, h;

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

  /* step */
  sampleNoise(&ts);
  h = ts - t;

  model->calculateDrift(t, y, a);
  model->calculateDiffusion(t, y, B);
  noalias(y) = y + h*a;
  ublas::axpy_prod(B, dWf.top(), y, false);

  /* update state */
  aux::vectorToArray(y, this->y); // don't use setState(), clears tf and dWf

  /* update time */
  t = ts;

  /* clean up future path */
  tf.pop(); // full step sample
  dWf.pop();

  /* post-condition */
  assert (tf.empty() || t < tf.top());
  assert (t <= upper);

  return t;
}

template <class DT, class DDT>
inline double indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::stepBack(
    double lower) {
  double t = NumericalSolver::stepBack(lower);

  return t;
}

template <class DT, class DDT>
inline int indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::calculateDerivativesForward(
    double ts, const double y[], double dydt[]) {
  /* not required by this implementation */
  assert (false);
  
  return GSL_FAILURE;
}

template <class DT, class DDT>
inline int indii::ml::sde::StochasticEulerMaruyama<DT,DDT>::calculateDerivativesBackward(
    double t, const double y[], double dydt[]) {
  /* not required by this implementation */
  assert (false);
  
  return GSL_FAILURE;
}

#endif

