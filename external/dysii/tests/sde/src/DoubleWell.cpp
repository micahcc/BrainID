#include "DoubleWell.hpp"

#define N 1

namespace aux = indii::ml::aux;

DoubleWell::DoubleWell() : indii::ml::sde::StochasticDifferentialModel<>(N) {
  //
}

DoubleWell::~DoubleWell() {
  //
}

indii::ml::aux::vector DoubleWell::calculateDrift(double ts,
    const indii::ml::aux::vector &y) {
  aux::vector result(N);
  result(0) = 4*y(0) * (THETA - y(0)*y(0));

  return result;
}

indii::ml::aux::matrix DoubleWell::calculateDiffusion(double ts,
    const indii::ml::aux::vector &y) {
  aux::matrix result(N,N);
  result(0,0) = SIGMA;

  return result;
}
