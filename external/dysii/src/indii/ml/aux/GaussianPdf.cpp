//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "GaussianPdf.hpp"
  #include "Random.hpp"
//#endif

#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_symmetric.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

#include <assert.h>

using namespace indii::ml::aux;

namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

GaussianPdf::GaussianPdf() : mu(0), sigma(0), L(0,0), sigmaI(0),
    sigmaIDiag(0), haveCholesky(false), haveInverse(false),
    haveDeterminant(false), haveZ(false), haveDensityOptimisations(false) {
  //
}

GaussianPdf::GaussianPdf(const vector& mu, const symmetric_matrix& sigma) :
    Pdf(mu.size()), mu(N), sigma(N), L(N,N), sigmaI(N), sigmaIDiag(N),
    haveCholesky(false), haveInverse(false), haveDeterminant(false),
    haveZ(false), haveDensityOptimisations(false) {
  /* pre-condition */
  assert(mu.size() == sigma.size1());

  /* expectation */ 
  setExpectation(mu);

  /* covariance */
  setCovariance(sigma);
}

GaussianPdf::GaussianPdf(unsigned int N) : Pdf(N), mu(N), sigma(N), L(N,N),
    sigmaI(N), sigmaIDiag(N), haveCholesky(false), haveInverse(false),
    haveDeterminant(false), haveZ(false), haveDensityOptimisations(false) {
  //
}

GaussianPdf& GaussianPdf::operator=(const GaussianPdf& o) {
  /* pre-condition */
  assert (o.N == N);

  haveCholesky = o.haveCholesky;
  haveInverse = o.haveInverse;
  haveDeterminant = o.haveDeterminant;
  haveZ = o.haveZ;
  haveDensityOptimisations = o.haveDensityOptimisations;

  mu = o.mu;
  sigma = o.sigma;
  if (haveCholesky) {
    L = o.L;
  }
  if (haveInverse) {
    sigmaI = o.sigmaI;
  }
  if (haveDeterminant) {
    sigmaDet = o.sigmaDet;
  }
  if (haveZ) {
    ZI = o.ZI;
  }
  if (haveDensityOptimisations) {
    isMuZero = o.isMuZero;
    isSigmaIDiagonal = o.isSigmaIDiagonal;
    if (isSigmaIDiagonal) {
      sigmaIDiag = o.sigmaIDiag;
    }
  }

  return *this;
}

GaussianPdf::~GaussianPdf() {
  //
}

void GaussianPdf::setDimensions(const unsigned int N, const bool preserve) {
  this->N = N;

  mu.resize(N, preserve);
  sigma.resize(N, preserve);
  L.resize(N, N, preserve);
  sigmaI.resize(N);
  sigmaIDiag.resize(N);

  dirty();
}

const vector& GaussianPdf::getExpectation() const {
  return mu;
}

const symmetric_matrix& GaussianPdf::getCovariance() const {
  return sigma;
}

const vector& GaussianPdf::getExpectation() {
  return mu;
}  

const symmetric_matrix& GaussianPdf::getCovariance() {
  return sigma;
}

void GaussianPdf::setExpectation(const vector& mu) {
  /* pre-condition */
  assert(mu.size() == N); // new same size as old

  this->mu = mu;
  dirty();
}

void GaussianPdf::setCovariance(const symmetric_matrix& sigma) {
  /* pre-condition */
  assert(sigma.size1() == N); // new same size as old

  this->sigma = sigma;
  dirty();
}

vector GaussianPdf::sample() {
  vector z(N);
  unsigned int i;

  if (!haveCholesky) {
    calculateCholesky();
    assert(haveCholesky);
  }

  /* \f$x\f$ sampled from \f$\mathcal{N}(\mathbf{0_N}, I_N)\f$ */
  for (i = 0; i < N; i++) {
    z(i) = Random::gaussian(0.0, 1.0);
  }

  /* \f$\mathbf{x} = L\mathbf{z} + \mu*/
  return mu + prod(L,z);
}

double GaussianPdf::densityAt(const vector& x) {
  if (!haveDensityOptimisations) {
    calculateDensityOptimisations();
    assert(haveDensityOptimisations);
  }

  double exponent;
  if (isMuZero && isSigmaIDiagonal) {
    exponent = inner_prod(x, element_prod(sigmaIDiag, x));
  } else {
    if (isMuZero) {
      exponent = inner_prod(x, prod(sigmaI, x));
    } else {
      aux::vector d(x - mu);
      if (isSigmaIDiagonal) {
        exponent = inner_prod(d, element_prod(sigmaIDiag, d));
      } else {
        exponent = inner_prod(d, prod(sigmaI, d));
      }
    }
  }
  
  double p = ZI * exp(-0.5 * exponent);
  if (isnan(p)) {
    p = 0.0;
  }
  
  /* post-condition */
  assert (p >= 0.0);
  
  return p;
}

void GaussianPdf::dirty() {
  haveCholesky = false;
  haveInverse = false;
  haveDeterminant = false;
  haveZ = false;
  haveDensityOptimisations = false;
}

double GaussianPdf::calculateDensity(const vector& x) {
  return densityAt(x);
}

void GaussianPdf::calculateCholesky() {
  symmetric_matrix tmp(sigma);
  int err;

  err = lapack::pptrf(tmp);
  assert (err == 0);
  noalias(L) = ublas::triangular_adaptor<symmetric_matrix, ublas::lower>(tmp);
  haveCholesky = true;
  
  /* post-condition */
  assert(haveCholesky);
}

void GaussianPdf::calculateInverse() {
  if (!haveCholesky) {
    calculateCholesky();
    assert(haveCholesky);
  }

  matrix X(L), I(N,N);
  I = identity_matrix(N);

  int err;
  err = lapack::potrs('L', X, I);
  assert (err == 0);
  noalias(sigmaI) = ublas::symmetric_adaptor<matrix, ublas::lower>(I);
  haveInverse = true;

  /* post-condition */
  assert(haveInverse);
}

void GaussianPdf::calculateDeterminant() {
  if (!haveCholesky) {
    calculateCholesky();
    assert(haveCholesky);
  }

  ublas::matrix_vector_range<lower_triangular_matrix> d = diag(L);
  assert(d.size() > 0);
  unsigned int i;
  double LDet = d(0);
  for (i = 1; i < d.size(); i++) {
    LDet *= d(i);
  }
  sigmaDet = LDet*LDet;
  haveDeterminant = true;

  /* post-condition */
  assert(haveDeterminant);
}

void GaussianPdf::calculateZ() {
  if (!haveDeterminant) {
    calculateDeterminant();
    assert(haveDeterminant);
  }
  ZI = 1.0 / sqrt(pow(2*M_PI,N) * sigmaDet);
  haveZ = true;

  /* post-condition */
  assert(haveZ);
}

void GaussianPdf::calculateDensityOptimisations() {
  unsigned int i, j;

  if (!haveDeterminant) {
    calculateDeterminant();
    assert(haveDeterminant);
  }
  if (!haveInverse) {
    calculateInverse();
    assert(haveInverse);
  }
  if (!haveZ) {
    calculateZ();
    assert(haveZ);
  }

  /* is expectation zero? */
  isMuZero = true;
  for (i = 0; i < N && isMuZero; i++) {
    isMuZero = isMuZero && mu(i) == 0.0;
  }

  /* is covariance diagonal? */
  isSigmaIDiagonal = true;
  for (j = 0; j < N && isSigmaIDiagonal; j++) {
    for (i = j+1; i < N && isSigmaIDiagonal; i++) {
      isSigmaIDiagonal = isSigmaIDiagonal && sigma(i,j) == 0.0;
      /* ^ better to check sigma rather than sigmaI elements here --
           probably supplied by user, making floating point equality
           comparisons reliable. */
    }
  }
  if (isSigmaIDiagonal) {
    noalias(sigmaIDiag) = diag(sigmaI);
  }

  haveDensityOptimisations = true;

  /* post-condition */
  assert (haveDensityOptimisations);
}

