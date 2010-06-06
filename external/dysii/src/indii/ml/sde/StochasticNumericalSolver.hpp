#ifndef INDII_ML_SDE_STOCHASTICNUMERICALSOLVER_HPP
#define INDII_ML_SDE_STOCHASTICNUMERICALSOLVER_HPP

#include "../ode/NumericalSolver.hpp"
#include "../aux/WienerProcess.hpp"

#include <stack>

namespace indii {
  namespace ml {
    namespace sde {
/**
 * Abstract numerical solver for a system of stochastic differential
 * equations.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 582 $
 * @date $Date: 2008-12-15 17:03:50 +0000 (Mon, 15 Dec 2008) $
 */
class StochasticNumericalSolver : public indii::ml::ode::NumericalSolver {
public:
  /**
   * Constructor.
   *
   * @param dimensions Dimensionality of the state space.
   * @param noises Dimensionality of the noise space.
   *
   * The time is initialised to zero, but the state is uninitialised
   * and should be set with setVariable() or setState().
   */
  StochasticNumericalSolver(const unsigned int dimensions,
      const unsigned int noises);

  /**
   * Constructor.
   *
   * @param y0 Initial state.
   * @param noises Dimensionality of the noise space.
   *
   * The time is initialised to zero and the state to that given.
   */
  StochasticNumericalSolver(const indii::ml::aux::vector& y0,
      const unsigned int noises);

  /**
   * Destructor.
   */
  virtual ~StochasticNumericalSolver();

protected:
  /**
   * Wiener process system noise.
   */
  indii::ml::aux::WienerProcess<double> W;

  /**
   * \f$t_1,t_2,\ldots > t\f$; future times.
   */
  std::stack<double> tf;

  /**
   * \f$\Delta\mathbf{W}(t_1),\Delta\mathbf{W}(t_2),\ldots\f$; Wiener process
   * increments at future times.
   */
  std::stack<indii::ml::aux::vector> dWf;

  /**
   * Sample Wiener process noise. This conditions on previous samples
   * of the Wiener trajectory at future times resulting from time
   * steps rejected by the adaptive step size control. This ensures
   * that steps are rejected only due to error bounds being exceeded
   * and not due to an improbable but perfectly valid Wiener
   * trajectory.
   *
   * @param ts \f$t_s\f$; the time at which to sample the noise. If this
   * is greater than the earliest time in the future path, will be adjusted
   * to this earliest time on return.
   *
   * @return True if a new Wiener increment was sampled, false if an
   * existing increment has been used.
   *
   * Let \f$t\f$ be the current time and \f$t_s\f$ the time at which
   * to sample the noise.
   *
   * Of the stored Wiener increments at times \f$t_1,t_2,\ldots >
   * t\f$, let \f$t_l\f$ be the greatest time less than or equal to
   * \f$t_s\f$, and \f$t_u\f$ the least time greater than
   * \f$t_s\f$. \f$\Delta\mathbf{W}(t_l)\f$ and
   * \f$\Delta\mathbf{W}(t_u)\f$ are the corresponding Wiener
   * increments.
   *
   * If \f$t_l\f$ does not exist, set \f$t_l \leftarrow t\f$.
   *
   * Then, if \f$t_s \neq t_l\f$, draw \f$\mathbf{\delta} \sim
   * \mathcal{N}(0,\sqrt{t_s-t_l})\f$. If \f$t_u\f$ does not exist,
   * set \f$\Delta\mathbf{W}(t_s) \leftarrow
   * \mathbf{\delta}\f$. Otherwise, set:
   *
   * \f{eqnarray*}
   * \Delta\mathbf{W}(t_s) &\leftarrow& \frac{t_s - t_l}{t_u -
   * t_l}\Delta\mathbf{W}(t_u) + \mathbf{\delta} \\
   * \Delta\mathbf{W}(t_u) &\leftarrow& \frac{t_u - t_s}{t_u -
   * t_l}\Delta\mathbf{W}(t_u) - \mathbf{\delta} \\
   * \f}
   *
   * and store.
   */
  bool sampleNoise(double* ts);

  virtual void reset();

};

    }
  }
}

#endif

