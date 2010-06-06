#include "DoubleWell.hpp"

#include "indii/ml/ode/EquilibriumSampler.hpp"
#include "indii/ml/sde/StochasticAdaptiveRungeKutta.hpp"

#include <iostream>
#include <fstream>

namespace aux = indii::ml::aux;

/**
 * @file EquilibriumSamplerHarness.cpp
 *
 * Test of EquilibriumSampler with DoubleWell model.
 *
 * Results are as follows:
 *
 * @image html EquilibriumSamplerHarness.png "Results"
 * @image latex EquilibriumSamplerHarness.eps "Results"
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
 * Number of samples to take.
 */
const unsigned int P = 1200;

/**
 * Burn time.
 */
const double BURN = 20.0;

/**
 * Interval between samples.
 */
const double INTERVAL = 1.0;

/**
 * Resolution of histogram.
 */
const unsigned int RES = 100;

/**
 * Lower bound on histogram.
 */
const double LOWER = -1.5;

/**
 * Upper bound on histogram.
 */
const double UPPER = 1.5;

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {
  unsigned int i, j;
  aux::vector y(M);
  double s;
  std::ofstream fout("results/EquilibriumSamplerHarness.out");

  DoubleWell model;
  indii::ml::sde::StochasticAdaptiveRungeKutta<> solver(&model);
  solver.setErrorBounds(1.0e-3, 1.0e-2);

  for (i = 0; i < N; i++) {
    y(0) = aux::Random::uniform(-1.0, 1.0);

    solver.setTime(0.0);
    solver.setState(y);
    solver.setStepSize(1.0e-4);

    indii::ml::ode::EquilibriumSampler stationary(&solver, BURN, INTERVAL);
    std::vector<unsigned int> counts(RES);
    for (j = 0; j < RES; j++) {
      counts[i] = 0;
    }

    for (j = 0; j < P; j++) {
      s = stationary.sample()(0);
      if (s > LOWER && s < UPPER) {
        counts[(int)floor((s - LOWER) / (UPPER - LOWER) * RES)]++;
      }
    }
    
    for (j = 0; j < RES; j++) {
      fout << LOWER + (UPPER - LOWER) * j / RES << '\t';
      fout << counts[j] << std::endl;
    }
    fout << std::endl;
  }

  return 0;
}

