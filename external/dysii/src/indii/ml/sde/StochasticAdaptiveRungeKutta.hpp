#ifndef INDII_ML_SDE_STOCHASTICADAPTIVERUNGEKUTTA_HPP
#define INDII_ML_SDE_STOCHASTICADAPTIVERUNGEKUTTA_HPP

#include "StochasticDifferentialModel.hpp"
#include "StochasticNumericalSolver.hpp"

namespace indii {
  namespace ml {
    namespace sde {
    
    template <class DT, class DDT>
    class StochasticAdaptiveRungeKuttaHelper;
    
/**
 * Stochastic Adaptive Runge-Kutta method for solving a system of
 * stochastic differential equations.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 566 $
 * @date $Date: 2008-09-13 22:41:27 +0100 (Sat, 13 Sep 2008) $
 *
 * @param DT Type of the diffusion matrix.
 * @param DDT Type of the diffusion partial derivative matrices.
 *
 * This class numerically solves StochasticDifferentialModel models
 * defining a system of stochastic differential equations using an
 * adaptive time step 4th/5th order (~2th/2.5th order for stochastic
 * systems) Runge-Kutta-Fehlberg method, as implemented in the
 * @ref GSL "GSL".
 *
 * The general usage idiom is as for
 * indii::ml::ode::AdaptiveRungeKutta. The class will automatically
 * keep track of Wiener process increments.
 *
 * @section StochasticAdaptiveRungeKutta_references References
 *
 * @anchor Wilkie2004
 * Wilkie, J. Numerical methods for stochastic differential
 * equations. <i>Physical Review E</i>, <b>2004</b>, <i>70</i>
 *
 * @anchor Sarkka2006
 * Särkkä, S. Recursive Bayesian Inference on Stochastic Differential
 * Equations. PhD thesis, <i>Helsinki University of Technology</i>,
 * <b>2006</b>.
 *
 * @todo This class is tightly coupled with the GSL and would benefit from
 * greater independence.
 */
template <class DT = indii::ml::aux::matrix,
    class DDT = indii::ml::aux::zero_matrix>
class StochasticAdaptiveRungeKutta :
    public indii::ml::sde::StochasticNumericalSolver {
    
    friend class indii::ml::sde::StochasticAdaptiveRungeKuttaHelper<DT,DDT>;
    
public:
  /**
   * Constructor.
   *
   * @param model Model to estimate.
   *
   * The time is initialised to zero, but the state is uninitialised
   * and should be set with setVariable() or setState().
   */
  StochasticAdaptiveRungeKutta(StochasticDifferentialModel<DT,DDT>* model);

  /**
   * Constructor.
   *
   * @param model Model to estimate.
   * @param y0 Initial state.
   *
   * The time is initialised to zero and the state to that given.
   */
  StochasticAdaptiveRungeKutta(StochasticDifferentialModel<DT,DDT>* model,
      const indii::ml::aux::vector& y0);

  /**
   * Destructor.
   */
  virtual ~StochasticAdaptiveRungeKutta();

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
  
  /**
   * Diffusion partial derivatives.
   */
  std::vector<DDT> dBdy;

  /**
   * GSL ordinary differential equations step type.
   */
  static const gsl_odeiv_step_type* gslStepType;

};

/* Omit these from documentation */
/// @cond STOCHASTICADAPTIVERUNGEKUTTAHELPER

/**
 * Helper class to facilitate specialisation for
 * DDT=indii::ml::aux::zero_matrix
 */
template <class DT, class DDT>
class StochasticAdaptiveRungeKuttaHelper {
public:
  static int calculateDerivativesForward(double t, const double y[],
      double dydt[], StochasticAdaptiveRungeKutta<DT,DDT>* sde);
};

/**
 * Partial specialisation for zero matrix diffusion derivatives (additive
 * noise), completely bypassing calculateDiffusionDerivatives() function
 * calls.
 */
template <class DT>
class StochasticAdaptiveRungeKuttaHelper<DT,indii::ml::aux::zero_matrix> {
public:
  static int calculateDerivativesForward(double t, const double y[],
      double dydt[],
      StochasticAdaptiveRungeKutta<DT,indii::ml::aux::zero_matrix>* sde);
};

/// @endcond

    }
  }
}

#include "boost/numeric/ublas/operation.hpp"
#include "boost/numeric/ublas/operation_sparse.hpp"

#include <assert.h>

#include <gsl/gsl_errno.h>

template <class DT, class DDT>
const gsl_odeiv_step_type* indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::gslStepType
    = gsl_odeiv_step_rkf45;

template <class DT, class DDT>
indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::StochasticAdaptiveRungeKutta(
    StochasticDifferentialModel<DT,DDT>* model) :
    StochasticNumericalSolver(model->getDimensions(),
    model->getNoiseDimensions()), model(model), a(model->getDimensions()),
    B(model->getDimensions(), model->getNoiseDimensions()) {
  unsigned int i;
  DDT dBdyi(model->getDimensions(), model->getNoiseDimensions());
  
  a.clear();
  B.clear();
  //dBdyi.clear();
  for (i = 0; i < model->getDimensions(); i++) {
    dBdy.push_back(dBdyi);
  }
  
  setStepType(gslStepType);
}

template <class DT, class DDT>
indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::StochasticAdaptiveRungeKutta(
    StochasticDifferentialModel<DT,DDT>* model,
    const indii::ml::aux::vector& y0) : StochasticNumericalSolver(y0,
    model->getNoiseDimensions()), model(model), a(model->getDimensions()),
    B(model->getDimensions(), model->getNoiseDimensions()) {
  /* pre-condition */
  assert (y0.size() == model->getDimensions());

  unsigned int i;
  DDT dBdyi(model->getDimensions(), model->getNoiseDimensions());
  
  a.clear();
  B.clear();
  //dBdyi.clear();
  for (i = 0; i < model->getDimensions(); i++) {
    dBdy.push_back(dBdyi);
  }

  setStepType(gslStepType);
}

template <class DT, class DDT>
indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::~StochasticAdaptiveRungeKutta() {
  //
}

template <class DT, class DDT>
double indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::step(
    double upper) {
  /* 
   * Implementation based on gsl_odeiv_evolve_apply() in 
   * ode-initval/evolve.c of the GSL (v1.11).
   */

  /* pre-condition */
  assert (upper > t);

  double ts, ts_new;
  double h, h_new;
  bool stepDecreased;
  int err;
  unsigned int i;

  /* make sure step size won't exceed upper bounds */
  if (maxStepSize > 0.0 && stepSize > maxStepSize) {
    stepSize = maxStepSize;
  }
  ts = t + stepSize;
  if (ts > upper) {
    ts = upper;
  }
  
  /* save initial state */
  memcpy(gslEvolve->y0, y, gslEvolve->dimension*sizeof(double));

  /* calculate initial derivative once */
  if (gslStep->type->can_use_dydt_in) {
    err = GSL_ODEIV_FN_EVAL(&gslForwardSystem, t, y, gslEvolve->dydt_in);
    assert (err == GSL_SUCCESS);
  }

  do {
    sampleNoise(&ts);
    h = ts - t;

    /* take proposed step */
    if (gslStep->type->can_use_dydt_in) {
      err = gsl_odeiv_step_apply(gslStep, t, h, y, gslEvolve->yerr,
          gslEvolve->dydt_in, gslEvolve->dydt_out, &gslForwardSystem);
    } else {
      err = gsl_odeiv_step_apply(gslStep, t, h, y, gslEvolve->yerr,
          NULL, gslEvolve->dydt_out, &gslForwardSystem);    
    }
    assert (err == GSL_SUCCESS);
    
    gslEvolve->last_step = h;
    
    /* adjust step size */
    /*
     * Usually we would call the following to adjust the step size
     * appropriately...
     */
    //err = gsl_odeiv_control_hadjust(gslControl, gslStep, y, gslEvolve->yerr,
    //    gslEvolve->dydt_out, &h);
    /*
     * ...This is translated to a simple function call like that below in
     * ode-initval/control.c of the GSL (v1.11). The order of the method
     * is passed on in this function call. Because we are working with a
     * stochastic system, the order of the method must be halved. We 
     * therefore expand out this function call directly here, halving
     * the order of the selected method.
     *
     * 03/08/08: The order is only used in adjusting the step size, so 
     * isn't that critical except for performance.
     * 05/08/08: If order is odd this rounds down, and in fact rkf45 is 
     * given as order 5, not 4... near enough though?
     */
    err = gslControl->type->hadjust(gslControl->state, gslStep->dimension,
        gslStep->type->order(gslStep->state) / 2, y, gslEvolve->yerr,
        gslEvolve->dydt_out, &h);
    
    stepDecreased = false;
    if (err == GSL_ODEIV_HADJ_DEC) {
      /* numerical checks */
      ts_new = t + h;
      if (ts_new > t && ts_new < ts) {
        h_new = ts_new - t;
        if (h_new > 0.0 && h_new < gslEvolve->last_step) {
          /* restore previous state and try again */
          ts = ts_new;
          memcpy(y, gslEvolve->y0, gslEvolve->dimension*sizeof(double));
          stepDecreased = true;
        } else {
          /* not ok, restore last step */
          h = gslEvolve->last_step;
        }
      } else {
        /* not ok, restore last step */
        h = gslEvolve->last_step;
      }
    }
  } while (stepDecreased); // repeat while step size too large

  /* update proposed step size */
  stepSize = h;

  /* update time */
  t = ts;

  /* clean up future path */
  tf.pop();
  dWf.pop();

  /* post-condition */
  assert (tf.empty() || t < tf.top());
  assert (t <= upper);

  return t;
}

template <class DT, class DDT>
inline double indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::stepBack(
    double lower) {
  double t = NumericalSolver::stepBack(lower);

  return t;
}

template <class DT, class DDT>
inline int indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::calculateDerivativesForward(
    double ts, const double y[], double dydt[]) {
  return StochasticAdaptiveRungeKuttaHelper<DT,DDT>::calculateDerivativesForward(
      ts, y, dydt, this);
}

template <class DT, class DDT>
inline int indii::ml::sde::StochasticAdaptiveRungeKutta<DT,DDT>::calculateDerivativesBackward(
    double t, const double y[], double dydt[]) {
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

template <class DT>
inline int
    indii::ml::sde::StochasticAdaptiveRungeKuttaHelper<DT,indii::ml::aux::zero_matrix>::calculateDerivativesForward(
    double ts, const double y[], double dydt[],
    StochasticAdaptiveRungeKutta<DT,indii::ml::aux::zero_matrix>* sde) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  StochasticDifferentialModel<DT,aux::zero_matrix>& model = *sde->model;
  aux::vector& a = sde->a;
  DT& B = sde->B;
    
  unsigned int i;
  const unsigned int N = model.getDimensions();
  const unsigned int W = model.getNoiseDimensions();
  const double t = sde->getTime();
  double delta;
  if (sde->tf.empty()) {
    delta = 0.0;
  } else {
    delta = sde->tf.top() - t;
  }

  aux::vector x(N);
  aux::arrayToVector(y, x);

  model.calculateDrift(ts, x, a);
  assert (a.size() == N);

  if (delta > 0.0) {
    model.calculateDiffusion(ts, x, B);
    assert (B.size1() == N && B.size2() == W);
    ublas::axpy_prod(B, sde->dWf.top() / delta, a, false);
  }
  aux::vectorToArray(a, dydt);

  return GSL_SUCCESS;
}

template <class DT, class DDT>
inline int
    indii::ml::sde::StochasticAdaptiveRungeKuttaHelper<DT,DDT>::calculateDerivativesForward(
    double ts, const double y[], double dydt[],
    StochasticAdaptiveRungeKutta<DT,DDT>* sde) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  StochasticDifferentialModel<DT,DDT>& model = *sde->model;
  aux::vector& a = sde->a;
  DT& B = sde->B;
  std::vector<DDT>& dBdy = sde->dBdy;

  unsigned int i;
  const unsigned int N = model.getDimensions();
  const unsigned int W = model.getNoiseDimensions();
  const double t = sde->getTime();
  double delta;
  if (sde->tf.empty()) {
    delta = 0.0;
  } else {
    delta = sde->tf.top() - t;
  }

  aux::vector x(N);
  aux::arrayToVector(y, x);

  model.calculateDrift(ts, x, a);
  assert (a.size() == N);

  if (delta > 0.0) {
    model.calculateDiffusion(ts, x, B);
    assert (B.size1() == N && B.size2() == W);

    model.calculateDiffusionDerivatives(ts, x, dBdy);
    assert (dBdy.size() == 0 || dBdy.size() == N);
    #ifdef NDEBUG
    if (dBdy.size() == N) {
      unsigned int j;
      for (j = 0; j < N; j++) {
        assert (dBdy[i].size1() == N && dBdy[i].size2() == W);
      }
    }
    #endif

    if (dBdy.size() > 0) {
      aux::vector b(N);
      b.clear();
      for (i = 0; i < N; i++) {
        ublas::axpy_prod(dBdy[i], row(B,i), b, false);
      }
      noalias(a) += 0.5 * b;
    }
    ublas::axpy_prod(B, sde->dWf.top() / delta, a, false);
  }
  aux::vectorToArray(a, dydt);

  return GSL_SUCCESS;
}

#endif

