#ifndef INDII_ML_ODE_EquilibriumSampler_HPP
#define INDII_ML_ODE_EquilibriumSampler_HPP

#include "NumericalSolver.hpp"

namespace indii {
  namespace ml {
    namespace ode {
/**
 * Samples from equilibrium distribution of a stationary process.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 510 $
 * @date $Date: 2008-08-07 14:16:46 +0100 (Thu, 07 Aug 2008) $
 *
 * Produces samples from the equilibrium distribution of a stationary
 * stochastic process using Markov Chain Monte Carlo (MCMC).
 */
class EquilibriumSampler {
public:
  /**
   * Constructor.
   *
   * @param solver Numerical solver.
   * @param burn Projected time until equilibrium distribution is reached
   * from initial state.
   * @param interval Time between successive samples once equilibrium
   * distribution has been reached, assuming samples are approximately
   * independent when separated by this interval.
   */
  EquilibriumSampler(NumericalSolver* solver, const double burn,
      const double interval);

  /**
   * Destructor.
   */
  virtual ~EquilibriumSampler();

  /**
   * Sample from the equilibrium distribution.
   *
   * @return Sample from the equilibrium distribution.
   *
   * On the first call, advances the system by the given burn time to reach
   * the equilibrium distribution, then returns the state of the system as
   * the sample. On subsequent calls, advances the system by the given
   * interval time and returns the state of the system as the sample.
   */
  indii::ml::aux::vector sample();
  
private:
  /**
   * Numerical solver.
   */
  NumericalSolver* solver;
  
  /**
   * Burn time.
   */
  const double burn;
  
  /**
   * Time between successive samples.
   */
  const double interval;

  /**
   * Number of samples taken.
   */
  unsigned int P;
  
};

    }
  }
}

#endif

