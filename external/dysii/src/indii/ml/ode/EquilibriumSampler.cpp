#include "EquilibriumSampler.hpp"

namespace aux = indii::ml::aux;

using namespace indii::ml::ode;

EquilibriumSampler::EquilibriumSampler(NumericalSolver* solver,
    const double burn, const double interval) : solver(solver), burn(burn),
    interval(interval), P(0) {
  //
}
    
EquilibriumSampler::~EquilibriumSampler() {
  //
}

indii::ml::aux::vector EquilibriumSampler::sample() {
  if (P == 0) {
    if (burn > 0.0) {
      solver->stepTo(solver->getTime() + burn);
    }
  } else if (interval > 0.0) {
    solver->stepTo(solver->getTime() + interval);
  }
  P++;
  
  return solver->getState();
}

