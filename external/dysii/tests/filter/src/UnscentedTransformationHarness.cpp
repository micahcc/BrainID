#include "indii/ml/filter/UnscentedTransformation.hpp"
#include "indii/ml/filter/UnscentedTransformationModel.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"

#include "gsl/gsl_statistics_double.h"
#include "gsl/gsl_rng.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * @file UnscentedTransformationHarness.cpp
 *
 * Tests of UnscentedTransformation.
 *
 * @section sanity Sanity test
 *
 * This is a sanity check of the unscented transformation. A random
 * multivariate Gaussian distribution is generated. The unscented
 * transformation is then used to propagate it through the trivial
 * #sanityModel.
 *
 * Results are as follows:
 *
 * @include UnscentedTransformationHarness_sanity.out
 *
 * @section linear Linear test
 *
 * Propagation through #linearModel.
 *
 * Results are as follows:
 *
 * @include UnscentedTransformationHarness_linear.out
 *
 * @section nonlinear Nonlinear test
 *
 * Propagation through #nonlinearModel.
 *
 * Results are as follows:
 *
 * @include UnscentedTransformationHarness_nonlinear.out
 */

/**
 * Dimensionality of the Gaussian.
 */
unsigned int M = 10;

/**
 * Number of samples to take for sampling output distribution.
 */
unsigned int N = 10000;

/**
 * Sanity check function \f$f(x) = x\f$ for testing.
 */
class SanityModel : public UnscentedTransformationModel<> {
public:
  virtual aux::vector propagate(const aux::vector& x, unsigned int delta = 0) {
    return x;
  }
} sanityModel;

/**
 * Linear function \f$f(x) = 3x\f$ for testing.
 */
class LinearModel : public UnscentedTransformationModel<> {
public:
  virtual aux::vector propagate(const aux::vector& x, unsigned int delta = 0) {
    return 3.0 * x;
  }
} linearModel;

/**
 * Nonlinear function \f$f(x) = x^2\f$ for testing.
 */
class NonlinearModel : public UnscentedTransformationModel<> {
public:
  virtual aux::vector propagate(const aux::vector& x, unsigned int delta = 0) {
    return element_prod(x, x);
  }
} nonlinearModel;

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {
  aux::vector mu(M);  // true mean
  aux::symmetric_matrix sigma(M);  // true covariance
  aux::lower_triangular_matrix tmp(M,M);  // to construct Cholesky decomp sigma
  aux::vector smu(M);  // sample mean
  aux::symmetric_matrix ssigma(M);  // sample covariance

  aux::vector sample(M);
  double data[M][N];
  unsigned int i, j;

  /* set up true distribution */
  gsl_rng_env_setup();
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, time(NULL));

  for (i = 0; i < M; i++) {
    mu(i) = gsl_rng_uniform(rng) * 10.0 - 5.0;
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j <= i; j++) {
      tmp(i,j) = gsl_rng_uniform(rng) * 5.0;
    }
  }
  noalias(sigma) = prod(tmp, trans(tmp)); // ensures cholesky decomposition
  aux::GaussianPdf x(mu, sigma);

  /* prepare for tests */
  aux::GaussianPdf y(x.getDimensions());
  UnscentedTransformation<> sanity(sanityModel);
  UnscentedTransformation<> linear(linearModel);
  UnscentedTransformation<> nonlinear(nonlinearModel);

  /* sanity test */
  ofstream fsanity("results/UnscentedTransformationHarness_sanity.out");
  fsanity << "mean(x)" << endl << mu << endl;
  fsanity << "cov(x)" << endl << sigma << endl;

  /* sample */
  for (i = 0; i < N; i++) {
    sample = sanityModel.propagate(x.sample());

    for (j = 0; j < M; j++) {
      data[j][i] = sample(j);
    }
  }
  for (i = 0; i < M; i++) {
    smu(i) = gsl_stats_mean(data[i], 1, N);
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      ssigma(i,j) = gsl_stats_covariance(data[i], 1, data[j], 1, N);
    }
  }
  fsanity << "sample mean(f(x))" << endl << smu << endl;
  fsanity << "sample cov(f(x))" << endl << ssigma << endl;

  /* unscented */
  y = sanity.transform(x, 1);
  fsanity << "unscented mean(f(x))" << endl << y.getExpectation() << endl;
  fsanity << "unscented cov(f(x))" << endl << y.getCovariance() << endl;

  fsanity.close();

  /* linear test */
  ofstream flinear("results/UnscentedTransformationHarness_linear.out");
  flinear << "mean(x)" << endl << mu << endl;
  flinear << "cov(x)" << endl << sigma << endl;

  /* sample */
  for (i = 0; i < N; i++) {
    sample = linearModel.propagate(x.sample());

    for (j = 0; j < M; j++) {
      data[j][i] = sample(j);
    }
  }
  for (i = 0; i < M; i++) {
    smu(i) = gsl_stats_mean(data[i], 1, N);
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      ssigma(i,j) = gsl_stats_covariance(data[i], 1, data[j], 1, N);
    }
  }
  flinear << "sample mean(f(x))" << endl << smu << endl;
  flinear << "sample covariance(f(x))" << endl << ssigma << endl;

  /* unscented */
  y = linear.transform(x, 1);
  flinear << "unscented mean(f(x))" << endl << y.getExpectation() << endl;
  flinear << "unscented cov(f(x))" << endl << y.getCovariance() << endl;

  flinear.close();

  /* nonlinear test */
  ofstream fnonlinear("results/UnscentedTransformationHarness_nonlinear.out");
  fnonlinear << "mean(x)" << endl << mu << endl;
  fnonlinear << "cov(x)" << endl << sigma << endl;

  /* sample */
  for (i = 0; i < N; i++) {
    sample = nonlinearModel.propagate(x.sample());

    for (j = 0; j < M; j++) {
      data[j][i] = sample(j);
    }
  }
  for (i = 0; i < M; i++) {
    smu(i) = gsl_stats_mean(data[i], 1, N);
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      ssigma(i,j) = gsl_stats_covariance(data[i], 1, data[j], 1, N);
    }
  }
  fnonlinear << "sample mean(f(x))" << endl << smu << endl;
  fnonlinear << "sample cov(f(x))" << endl << ssigma << endl;

  /* unscented */
  y = nonlinear.transform(x, 1);
  fnonlinear << "unscented mean(f(x))" << endl << y.getExpectation() << endl;
  fnonlinear << "unscented cov(f(x))" << endl << y.getCovariance() << endl;
  fnonlinear.close();
}
