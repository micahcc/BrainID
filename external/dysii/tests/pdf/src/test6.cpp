#include "indii/ml/aux/GaussianMixturePdf.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/Random.hpp"

#include <gsl/gsl_statistics_double.h>

#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test6.cpp
 *
 * Test of GaussianMixturePdf.
 *
 * This test creates a random multivariate Gaussian mixture. It then
 * samples from this mixture and compares the mean and covariance of
 * the original mixture with the mean and covariance of the sample
 * set.
 *
 * Results are as follows:
 *
 * @include test6.out
 */

/**
 * Dimensionality of the Gaussian mixture.
 */
unsigned int M = 10;

/**
 * Number of components in the Gaussian mixture.
 */
unsigned int COMPONENTS = 12;

/**
 * Number of samples to take.
 */
unsigned int N = 100000;

/**
 * Create random Gaussian distribution.
 *
 * @param M Dimensionality of the Gaussian.
 * @param minMean Minimum value of any component of the mean.
 * @param maxMean Maximum value of any component of the mean.
 * @param minCov Minimum value of any component of the covariance.
 * @param maxCov Maximum value of any component of the covariance.
 *
 * @return Gaussian with given dimensionality, with mean and
 * covariance randomly generated uniformly from within the given
 * bounds.
 */
aux::GaussianPdf createRandomGaussian(const unsigned int M,
    const double minMean = -5.0, const double maxMean = 5.0,
    const double minCov = 0.0, const double maxCov = 5.0) {
  aux::vector mu(M);
  aux::symmetric_matrix sigma(M);

  unsigned int i, j;

  /* mean */
  for (i = 0; i < M; i++) {
    mu(i) = aux::Random::uniform(minMean, maxMean);
  }

  /* covariance */
  for (i = 0; i < M; i++) {
    for (j = 0; j <= i; j++) {
      sigma(i,j) = aux::Random::uniform(sqrt(minCov) / M, sqrt(maxCov) / M);
    }
  }
  sigma = prod(sigma, trans(sigma)); // ensures cholesky decomposable

  return aux::GaussianPdf(mu, sigma);
}

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {
  aux::vector sample(M);
  double data[M][N];
  unsigned int i, j;
  aux::vector smu(M);  // sample mean
  aux::symmetric_matrix ssigma(M);  // sample covariance

  /* create distribution */
  aux::GaussianMixturePdf pdf(M);
  for (i = 0; i < COMPONENTS; i++) {
    pdf.add(createRandomGaussian(M), aux::Random::uniform(0.0,1.0));
  }

  /* sample from distribution */
  for (i = 0; i < N; i++) {
    sample = pdf.sample();

    for (j = 0; j < M; j++) {
      data[j][i] = sample(j);
    }
  }

  /* calculate mean and variance of samples */
  for (i = 0; i < M; i++) {
    smu(i) = gsl_stats_mean(data[i], 1, N);
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      ssigma(i,j) = gsl_stats_covariance(data[i], 1, data[j], 1, N);
    }
  }

  cout << "True mean" << endl << pdf.getExpectation() << endl;
  cout << "True covariance" << endl << pdf.getCovariance() << endl;
  cout << "Sample mean" << endl << smu << endl;
  cout << "Sample covariance" << endl << ssigma << endl;
}
