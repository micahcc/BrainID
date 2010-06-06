#ifndef INDII_ML_ODE_ADAPTIVERUNGEKUTTA_HPP
#define INDII_ML_ODE_ADAPTIVERUNGEKUTTA_HPP

#include "NumericalSolver.hpp"
#include "DifferentialModel.hpp"
#include "../aux/vector.hpp"

#include <gsl/gsl_odeiv.h>

namespace indii {
  namespace ml {
    namespace ode {

/**
 * Adaptive Runge-Kutta method for solving a system of ordinary
 * differential equations.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 448 $
 * @date $Date: 2008-05-01 17:11:33 +0100 (Thu, 01 May 2008) $
 *
 * This class numerically solves DifferentialModel models defining a
 * system of ordinary differential equations using an adaptive time
 * step 4th/5th order Runge--Kutta--Fehlberg method as implemented in
 * the @ref GSL "GSL".
 *
 * The general usage idiom is as follows. First begin by writing your
 * own class deriving from DifferentialModel to describe your model,
 * and instantiating an object of this class:
 *
 * @code
 * MyDifferentialModel model;
 * @endcode
 *
 * Construct the initial state of the system:
 *
 * @code
 * indii::ml::aux::vector y0(4);
 * y0(0) = 1.0;
 * y0(1) = 0.0;
 * y0(2) = 0.0;
 * y0(3) = 2.0;
 * @endcode
 * 
 * Create an AdaptiveRungeKutta object, passing in the model to solve
 * and the initial state:
 *
 * @code
 * AdaptiveRungeKutta solver(&model, y0);
 * @endcode
 *
 * At this stage setErrorBounds() and setStepSize() may be used to
 * manipulate the Runge-Kutta to be appropriate for the model,
 * balancing speed versus accuracy. The defaults are often sufficient,
 * however.
 *
 * Now use step() to step through the system. The time step will vary
 * to maintain error bounds, so it is necessary to keep track of the
 * time after each step.
 *
 * @code
 * double end = 60.0;
 * double t = 0.0;
 * while (t < end) {
 *   t = solver.step(end);
 *   std::cout << solver.getState() << std::endl;
 * }
 * @endcode
 *
 * @c t is guaranteed to reach @c end at some point, although the
 * number of steps that this will take depends on the model, initial
 * state and error bounds.
 * 
 * If you are only interested in the state of the system at a given
 * time, or wish to use fixed time steps, use stepTo() to jump
 * straight to the required time. Internally, stepTo() essentially
 * just performs the above loop for you, but its use can be more
 * convenient.
 *
 * For known discontinuities, it is usual to estimate the function
 * piecewise, stepping from start to finish through a continuous
 * piece, then calling setDiscontinuity() before proceeding onto the
 * next.
 *
 * @section AdaptiveRungeKutta_references References
 *
 * @anchor GSL
 * The GNU Scientific Library (GSL). http://www.gnu.org/software/gsl/.
 */
class AdaptiveRungeKutta : public NumericalSolver {
public:
  /**
   * Constructor.
   *
   * @param model Model to estimate.
   *
   * The time is initialised to zero, but the state is uninitialised
   * and should be set with setVariable() or setState().
   */
  AdaptiveRungeKutta(DifferentialModel* model);

  /**
   * Constructor.
   *
   * @param model Model to estimate.
   * @param y0 Initial state.
   *
   * The time is initialised to zero and the state to that given.
   */
  AdaptiveRungeKutta(DifferentialModel* model,
      const indii::ml::aux::vector& y0);

  /**
   * Destructor.
   */
  virtual ~AdaptiveRungeKutta();

  virtual int calculateDerivativesForward(double t, const double y[],
      double dydt[]);

  virtual int calculateDerivativesBackward(double t, const double y[],
      double dydt[]);

private:
  /**
   * Model.
   */
  DifferentialModel* model;

  /**
   * GSL ordinary differential equations step type.
   */
  static const gsl_odeiv_step_type* gslStepType;

};

    }
  }
}

#endif

