#ifndef INDII_ML_ODE_AUTOCORRELATOR_HPP
#define INDII_ML_ODE_AUTOCORRELATOR_HPP

#include "NumericalSolver.hpp"
#include "../aux/matrix.hpp"

namespace indii {
  namespace ml {
    namespace ode {
/**
 * Auto-correlator.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 574 $
 * @date $Date: 2008-10-04 14:14:57 +0100 (Sat, 04 Oct 2008) $
 *
 * Calculates the autocorrelation of a Markov process for a particular time
 * step \f$\Delta t\f$:
 *
 * \f[
 *   R_s(\Delta t) = \left(\sum_{n = 1}^{s}\mathbf{y}_{n-1}\mathbf{y}_n^T -
 *   \hat{\mathbf{\mu}}_s\hat{\mathbf{\mu}}_s^T\right) \hat{\Sigma}_s^{-1}
 *   \,,
 * \f]
 *
 * where \f$s\f$ is the current step, each \f$\mathbf{y}_n\f$ is the state
 * of the system at step \f$n\f$ (time \f$n\Delta t\f$), and
 * \f$\hat{\mathbf{\mu}}_s\f$ and \f$\hat{\Sigma}_s\f$ are the sample mean
 * and covariance of \f$\mathbf{y}_0,\ldots,\mathbf{y}_s\f$, respectively.
 *
 * The autocovariance is given by the autocorrelation without the
 * normalisation term.
 *
 * @section Usage
 *
 * Firstly construct a NumericalSolver for simulating a trajectory from the
 * model of interest. Pass this into the constructor of the
 * AutoCorrelator object.
 *
 * Call step() to advance the system by a number of steps, adding each new
 * point to the autocorrelation calculation. The return value of step()
 * indicates whether the calculation has converged. Note that this
 * convergence check compares the autocorrelations before and after the call
 * to step(), so that multiple calls are necessary for the return value to be
 * meaningful.
 */
class AutoCorrelator {
public:
  /**
   * Constructor.
   *
   * @param solver Numerical solver.
   * @param delta \f$\Delta t\f$; time step.
   */
  AutoCorrelator(NumericalSolver* solver, const double delta);

  /**
   * Destructor.
   */
  virtual ~AutoCorrelator();

  /**
   * Set the error bounds for the convergence criterion.
   *
   * @param maxAbsoluteError The maximum permitted absolute error.
   */
  void setErrorBounds(double maxAbsoluteError = 1e-3);

  /**
   * Get the autocorrelation as calculated up to the current time.
   *
   * @return Autocorrelation as calculated up to the current time.
   */
  const indii::ml::aux::matrix& getAutoCorrelation();
  
  /**
   * Get the autocovariance as calculated up to the current time.
   *
   * @return Autocovariance as calculated up to the current time.
   */
  const indii::ml::aux::matrix& getAutoCovariance();
  
  /**
   * Step.
   *
   * @param steps \f$\delta s\f$; number of steps to take.
   *
   * @return True if the calculation has converged.
   *
   * Advances system in time by \f$\Delta s \Delta t\f$ and adds the new 
   * point to the autocorrelation calculation.
   *
   * Convergence is checked in the sense:
   *
   * \f[
   *   \|R_s(\Delta t) - R_{s+\Delta s}(\Delta t)\| < \epsilon +
   *   \xi\|R_s(\Delta t) - R_{s+\Delta s}(\Delta t)\|\,,
   * \f]
   *
   * where \f$\epsilon\f$ is the maximum permitted absolute error, and
   * \f$\xi\f$ the maximum permitted relative error.
   */
  bool step(const unsigned int steps);
  
private:
  /**
   * Numerical solver.
   */
  NumericalSolver* solver;
  
  /**
   * \f$\Delta t\f$; time step.
   */
  const double delta;
  
  /**
   * \f$t\f$; current time step.
   */
  unsigned int s;

  /**
   * \f$s\mathbf{\mu}_t\f$; mean at current time.
   */
  indii::ml::aux::vector mu;
  
  /**
   * \f$\Sigma_t\f$; covariance at current time, no mean correction.
   */
  indii::ml::aux::symmetric_matrix sigma;
  
  /**
   * \f$\frac{1}{s}\sum_{n = 1}{s}\mathbf{y}_{n-1}\mathbf{y}_n^T\f$;
   * cross correlation numerator sum at current time.
   */
  indii::ml::aux::matrix cross;

  /**
   * \f$R_s(\Delta t)\f$; autocorrelation at the current time.
   */
  indii::ml::aux::matrix R;
  
  /**
   * Autocovariance at the current time.
   */
  indii::ml::aux::matrix P;
  
  /**
   * \f$\epsilon\f$; absolute error bound.
   */
  double maxAbsoluteError;
  
};

    }
  }
}

#endif

