#include "AutoCorrelator.hpp"

namespace aux = indii::ml::aux;

using namespace indii::ml::ode;

AutoCorrelator::AutoCorrelator(NumericalSolver* solver,
    const double delta) :
    solver(solver),
    delta(delta),
    s(0),
    mu(solver->getDimensions()),
    sigma(solver->getDimensions()),
    cross(solver->getDimensions(), solver->getDimensions()),
    R(solver->getDimensions(), solver->getDimensions()),
    P(solver->getDimensions(), solver->getDimensions()) {
  /* pre-condition */
  assert (delta > 0.0);
  
  aux::vector y(solver->getState());
  noalias(mu) = y;
  noalias(sigma) = outer_prod(y,y);
  cross.clear();
  R.clear();
  P.clear();
  
  setErrorBounds();
}

AutoCorrelator::~AutoCorrelator() {
  //
}

const aux::matrix& AutoCorrelator::getAutoCorrelation() {
  return R;
}

const aux::matrix& AutoCorrelator::getAutoCovariance() {
  return P;
}

void AutoCorrelator::setErrorBounds(double maxAbsoluteError) {
  this->maxAbsoluteError = maxAbsoluteError;
}

bool AutoCorrelator::step(const unsigned int steps) {
  unsigned int i;
  aux::matrix R0(R); // initial autocorrelation
  aux::vector x(solver->getDimensions()), y(solver->getDimensions());

  /* step */
  for (i = 0; i < steps; i++) {
    noalias(x) = solver->getState();
    solver->stepTo(solver->getTime() + delta);
    noalias(y) = solver->getState();

    s++;
    noalias(mu) += y;
    noalias(sigma) += outer_prod(y,y);
    noalias(cross) += outer_prod(x,y);
  }

  /* calculate new autocovariance and autocorrelation */
  aux::vector mean(mu / (s + 1.0));
  aux::symmetric_matrix mean2(outer_prod(mean,mean));
  noalias(P) = cross / s - mean2;
  
  aux::matrix tmp(sigma / s - mean2);
  aux::matrix sigmaI(tmp.size1(), tmp.size2());
  aux::inv(tmp, sigmaI);
  noalias(R) = prod(P, sigmaI);

  /* check convergence */
  double e = norm_frobenius(R - R0);
  
  return e < maxAbsoluteError;
}

