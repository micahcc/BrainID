#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/Random.hpp"

#include <gsl/gsl_statistics_double.h>

#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test2.cpp
 *
 * Multivariate test of GaussianPdf.
 *
 * This test creates a multivariate Gaussian with a random mean and
 * covariance. It then samples from this distribution and calculates
 * the sample mean and variance for comparison.
 *
 * Results are as follows:
 *
 * @include test2.out
 */

/**
 * Dimensionality of the Gaussian.
 */
unsigned int M = 10;

/**
 * Number of samples to take.
 */
unsigned int N = 100000;

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

  /* set up distribution */
  for (i = 0; i < M; i++) {
    mu(i) = aux::Random::uniform(-5.0, 5.0);
  }

  for (i = 0; i < M; i++) {
    for (j = 0; j <= i; j++) {
      tmp(i,j) = aux::Random::uniform(0.0, 5.0);
    }
  }
  noalias(sigma) = prod(tmp, trans(tmp)); // ensures cholesky decomposition
  aux::GaussianPdf pdf(mu, sigma);

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

  cout << "True mean" << endl << mu << endl;
  cout << "True covariance" << endl << sigma << endl;
  cout << "Sample mean" << endl << smu << endl;
  cout << "Sample covariance" << endl << ssigma << endl;
}
