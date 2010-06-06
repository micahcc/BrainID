#include "UniformPdf.hpp"
#include "Random.hpp"

using namespace indii::ml::aux;

UniformPdf::UniformPdf() : p(0.0), lower(0), upper(0), mu(0),
    sigma(0), haveMu(false), haveSigma(false) {
  //
}

UniformPdf::UniformPdf(const vector& lower, const vector& upper) :
    Pdf(lower.size()), lower(lower), upper(upper), mu(lower.size()),
    sigma(lower.size()), haveMu(false), haveSigma(false) {
  /* pre-condition */
  assert(lower.size() == upper.size());

  calculateDensity();
}

UniformPdf& UniformPdf::operator=(const UniformPdf& o) {
  /* pre-condition */
  assert (o.N == N);

  haveMu = o.haveMu;
  haveSigma = o.haveSigma;
  
  p = o.p;
  lower = o.lower;
  upper = o.upper;
  if (haveMu) {
    mu = o.mu;
  }
  if (haveSigma) {
    sigma = o.sigma;
  }

  return *this;
}

UniformPdf::~UniformPdf() {
  //
}

void UniformPdf::setDimensions(const unsigned int N, const bool preserve) {
  this->N = N;

  lower.resize(N, preserve);
  upper.resize(N, preserve);
  mu.resize(N, preserve);
  sigma.resize(N, preserve);
  
  calculateDensity();
}

const vector& UniformPdf::getExpectation() {
  if (!haveMu) {
    calculateExpectation();
  }
  return mu;
}

const symmetric_matrix& UniformPdf::getCovariance() {
  if (!haveSigma) {
    calculateCovariance();
  }
  return sigma;
}

vector UniformPdf::sample() {
  vector s(N);
  unsigned int i;
  for (i = 0; i < N; i++) {
    s(i) = Random::uniform(lower(i), upper(i));
  }
  
  return s;
}

double UniformPdf::densityAt(const vector& x) {
  bool inside = true;
  unsigned int i;
  
  for (i = 0; i < N && inside; i++) {
    inside = inside && lower(i) <= x(i) && x(i) < upper(i);
  }
  return inside ? p : 0.0;
}

void UniformPdf::calculateExpectation() {
  noalias(mu) = (lower + upper) / 2;
  haveMu = true;
  
  /* post-condition */
  assert (haveMu);
}

void UniformPdf::calculateCovariance() {
  vector diff(upper - lower);
  noalias(sigma) = outer_prod(diff, diff) / 12;
  haveSigma = true;
  
  /* post-condition */
  assert (haveSigma);
}

void UniformPdf::calculateDensity() {
  double volume = 1.0;
  vector::iterator iter, end;
  vector diff(upper - lower);
  
  iter = diff.begin();
  end = diff.end();
  while (iter != end) {
    volume *= *iter;
    iter++;
  }
  
  p = 1.0 / volume;
}

