#ifndef INDII_ML_ODE_NUMERICALSOLVER_HPP
#define INDII_ML_ODE_NUMERICALSOLVER_HPP

#include "../aux/vector.hpp"

#include <gsl/gsl_odeiv.h>

namespace indii {
  namespace ml {
    namespace ode {

/**
 * Type of functions for calculating derivatives.
 */
typedef int df_t(double t, const double y[], double dydt[], void* params);

/**
 * Abstract numerical solver for a system of differential equations.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 521 $
 * @date $Date: 2008-08-16 16:58:51 +0100 (Sat, 16 Aug 2008) $
 */
class NumericalSolver {
public:
  /**
   * Constructor.
   *
   * @param dimensions Dimensionality of the state space.
   *
   * The time is initialised to zero, but the state is uninitialised
   * and should be set with setVariable() or setState().
   */
  NumericalSolver(const unsigned int dimensions);

  /**
   * Constructor.
   *
   * @param y0 Initial state.
   *
   * The time is initialised to zero and the state to that given.
   */
  NumericalSolver(const indii::ml::aux::vector& y0);

  /**
   * Destructor.
   */
  virtual ~NumericalSolver();

  /**
   * Get dimensionality of the state space.
   */
  unsigned int getDimensions();

  /**
   * Get the current time.
   */
  double getTime();

  /**
   * Set the current time.
   *
   * @param t Time.
   *
   * @see setVariable() or setState() to update the state for the new
   * time also.
   */
  void setTime(const double t);

  /**
   * Get the value of a state variable at the current time.
   *
   * @param index Index of the state variable to retrieve.
   *
   * @return The value of the state variable at the current time.
   */
  double getVariable(const unsigned int index);

  /**
   * Set the value of a state variable.
   *
   * @param index Index of the state variable to set.
   * @param value The value to which to set the state variable.
   */
  void setVariable(const unsigned int index, const double value);

  /**
   * Get the state at the current time.
   *
   * @return Current state of the system.
   */
  indii::ml::aux::vector getState();

  /**
   * Set the state at the current time.
   *
   * @param y New state of the system.
   */
  void setState(const indii::ml::aux::vector& y);

  /**
   * Advance the system one time step. The size of the step is chosen
   * to be optimal given the provided error bounds. State variables
   * are updated to this new time after completion of the step.
   *
   * @param upper Upper bound on time. This must be greater than the
   * current time. The current time is guaranteed not to exceed this
   * value after the step is complete. It may equal this time, and
   * indeed if step() is continuously called with the same upper bound
   * it will eventually do so.
   *
   * @return The new current time.
   */
  virtual double step(double upper);

  /**
   * Advance the system to a particular time.
   *
   * This convenience method internally calls step() as many times as
   * necessary to advance the system to the given time. This is useful
   * if fixed time steps are required rather than the variable steps
   * taken by step().
   *
   * @param to The time to which to advance the system. This must be greater
   * than the current time. The current time is guaranteed to match this value
   * at the end of the call.
   */
  void stepTo(double to);

  /**
   * Rewind the system one time step. This works in the same fashion
   * at step(), except that the step is backwards in time.
   *
   * @param lower Lower bound on time. This must be less than the
   * current time. The current time is guaranteed not to be below this
   * value after the step is complete. It may equal this time, and
   * indeed if stepBack() is continuously called with the same lower
   * bound it will eventually do so.
   *
   * @return The new current time.
   */
  virtual double stepBack(double lower);

  /**
   * Rewind the system to a specific time.
   *
   * This convenience method internally calls stepBack() as many times
   * as necessary to rewind the system to a given point in time. This
   * is useful if fixed time steps are required rather than the
   * variable steps taken by stepBack().
   *
   * @param to The time to which to rewind the system. This must be
   * less than the current time. The current time is guaranteed to
   * match this value at the end of the call.
   */
  void stepBackTo(double to);

  /**
   * Set the error bounds. Smaller error bounds produce more accurate
   * estimates, but reduce the size of time steps, so that more steps
   * are required in order to estimate functions over the same length
   * of time.
   *
   * @param maxAbsoluteError The maximum permitted absolute error of
   * estimated values compared to real values.
   * @param maxRelativeError The maximum permitted relative error of
   * estimated values compared to real values.
   */
  void setErrorBounds(double maxAbsoluteError = 1e-6,
      double maxRelativeError = 1e-6);

  /**
   * Set the proposed size for the next time step. This is useful for
   * handling discontinuities, where step() may be called to advance
   * to the discontinuity, then setStepSize() used to propose a small
   * step size ensuring that the discontinuity is tiptoed across
   * rather than leapt.
   *
   * It should be called immediately before any call to step() or
   * stepTo(). Any other methods called in between may themselves adjust the
   * proposed step size.
   *
   * @param stepSize The proposed time step size for the next call to
   * step().
   *
   * Note that this sets only the proposed step size -- it will be
   * optimised, within some constraints, relative to the permitted
   * error bounds.
   */
  void setStepSize(double stepSize);

  /**
   * Get the proposed size for the next time step.
   *
   * @return The proposed time step size for the next call to step().
   */
  double getStepSize();

  /**
   * Set the suggested step size. The step size is set to this
   * immediately, and subsequently after any calls to
   * setDiscontinuity().
   *
   * @param stepSize The suggested step size.
   *
   * Note that this sets only the proposed step size -- it will be
   * optimised, within some constraints, relative to the permitted
   * error bounds.
   */
  void setSuggestedStepSize(double stepSize = 1.0e-7);

  /**
   * Get the suggested step size.
   *
   * @return The suggested step size.
   */
  double getSuggestedStepSize();

  /**
   * Set the maximum step size.
   *
   * @param stepSize The maximum step size.
   *
   * This is useful for bounding the step size on schemes which may propose
   * steps that, for some models, produce inf or nan derivative calculations.
   * Empirically, we particularly observe this with implicit schemes. By
   * default the step size is not bound.
   */
  void setMaxStepSize(double stepSize = 0.0);

  /**
   * Get the maximum step size.
   *
   * @return The maximum step size.
   */
  double getMaxStepSize();

  /**
   * Indicates a discontinuity at the current time. Internally, this simply
   * calls setStepSize() with the suggested step size.
   */
  void setDiscontinuity();

  /**
   * Set the stepping method.
   *
   * @param stepType The stepping method.
   *
   * This allows modification of the underlying numerical scheme used by
   * the solver, as specified by the GSL. For advanced users only.
   */
  void setStepType(const gsl_odeiv_step_type* stepType);

  /**
   * Calculate derivatives for forwards step.
   */
  virtual int calculateDerivativesForward(double t, const double y[],
      double dydt[]) = 0;

  /**
   * Calculate derivatives for backwards step.
   */
  virtual int calculateDerivativesBackward(double t, const double y[],
      double dydt[]) = 0;

protected:
  /**
   * Dimensionality of the system of differential equations.
   */
  const size_t dimensions;

  /**
   * \f$t\f$; the current time
   */
  double t;

  /**
   * Array of state variables of the system.
   */
  double* y;

  /**
   * Base time used for backward steps.
   */
  double base;

  /**
   * Proposed size for the next time step.
   */
  double stepSize;

  /**
   * Maximum step size.
   */
  double maxStepSize;

  /**
   * Suggested step size.
   */
  double suggestedStepSize;

  /**
   * GSL ordinary differential equations step structure.
   */
  gsl_odeiv_step* gslStep;

  /**
   * GSL ordinary differential equations control structure.
   */
  gsl_odeiv_control* gslControl;

  /**
   * GSL ordinary differential equations evolution structure.
   */
  gsl_odeiv_evolve* gslEvolve;
 
  /**
   * GSL system of ordinary differential equations for forward steps
   * in time.
   */
  gsl_odeiv_system gslForwardSystem;

  /**
   * GSL system of ordinary differential equations for backward steps
   * in time.
   */
  gsl_odeiv_system gslBackwardSystem;

  /**
   * Initialise the system. Should be called before any calls to
   * step() or stepTo().
   */
  void init();

  /**
   * Terminate the system. Should be called after work with the system
   * has been finished. Will be called by the destructor if it hasn't
   * been already.
   */
  void terminate();

  /**
   * Reset the model. This is used internally to reset GSL structures when the
   * next step will not be a continuation of the last, usually when the time
   * or state is changed with a call to setTime(), setVariable() or
   * setState().
   */
  virtual void reset();

private:
  /**
   * Function for calculating derivatives when moving forward in time.
   *
   * Required by gsl_odeiv_system struct of GSL, which is in turn
   * passed to the gsl_odeiv_evolve_apply() function. Acts as a
   * wrapper to call the calculateForwardDerivatives() function of the
   * particular NumericalSolver object.
   *
   * @see gsl_odeiv_system data type of GSL.
   * @see gsl_odeiv_evolve_apply() of GSL.
   */
  static int gslForwardFunction(double t, const double y[], double dydt[],
      void* params);

  /**
   * Function for calculating derivatives when moving backward in time.
   * Acts as a wrapper to call the calculateBackwardDerivatives() function
   * of the particular NumericalSolver object.
   *
   * @see gslForwardFunction()
   */
  static int gslBackwardFunction(double t, const double y[], double dydt[],
      void* params);

};


    }
  }
}

inline unsigned int indii::ml::ode::NumericalSolver::getDimensions() {
  return dimensions;
}

inline double indii::ml::ode::NumericalSolver::getTime() {
  return t;
}

inline double indii::ml::ode::NumericalSolver::getVariable(
    const unsigned int index) {
  /* pre-condition */
  assert (index < dimensions);

  return y[index];
}

inline indii::ml::aux::vector indii::ml::ode::NumericalSolver::getState() {
  indii::ml::aux::vector y(dimensions);
  unsigned int i;

  for (i = 0; i < dimensions; i++) {
    y(i) = this->y[i];
  }
  return y;
}

inline double indii::ml::ode::NumericalSolver::getStepSize() {
  return stepSize;
}

inline double indii::ml::ode::NumericalSolver::getSuggestedStepSize() {
  return suggestedStepSize;
}

inline double indii::ml::ode::NumericalSolver::getMaxStepSize() {
  return maxStepSize;
}

#endif

