#include "DoubleWell.hpp"

#include "indii/ml/ode/AutoCorrelator.hpp"
#include "indii/ml/sde/StochasticAdaptiveRungeKutta.hpp"

#include <iostream>
#include <fstream>

namespace aux = indii::ml::aux;

/**
 * @file AutoCorrelatorHarness.cpp
 *
 * Test of AutoCorrelator with DoubleWell model.
 *
 * This test calculates the autocorrelation of DoubleWell using
 * indii::ml::ode::AutoCorrelator.
 *
 * Results are as follows:
 *
 * @image html AutoCorrelatorHarness.png "Results"
 * @image latex AutoCorrelatorHarness.eps "Results"
 */

/**
 * Dimensionality of the process.
 */
const unsigned int M = 1;

/**
 * Number of sample trajectories.
 */
const unsigned int N = 1;

/**
 * Time length of each trajectory.
 */
const double LENGTH = 1200.0;

/**
 * Autocorrelation step.
 */
const double DELTA = 1.0;

/**
 * Autocorrelation steps between convergence checks.
 */
const unsigned int STEPS = 10;

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {
  unsigned int i;
  aux::vector y(M);
  bool hasConverged;
  std::ofstream fout("results/AutoCorrelatorHarness.out");

  DoubleWell model;
  indii::ml::sde::StochasticAdaptiveRungeKutta<> solver(&model);
  solver.setErrorBounds(1.0e-3, 1.0e-2);

  for (i = 0; i < N; i++) {
    y(0) = aux::Random::uniform(-1.0, 1.0);

    solver.setTime(0.0);
    solver.setState(y);
    solver.setStepSize(1.0e-4);

    indii::ml::ode::AutoCorrelator autocor(&solver, DELTA);

    while (solver.getTime() < LENGTH) {
      hasConverged = autocor.step(STEPS);    

      fout << solver.getTime() << '\t';
      fout << autocor.getAutoCorrelation()(0,0) << '\t';
      if (hasConverged) {
        fout << 1;
      } else {
        fout << 0;
      }
      fout << std::endl;
    }
    fout << std::endl;
  }

  return 0;
}

