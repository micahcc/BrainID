#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/Random.hpp"

#include "gsl/gsl_statistics_double.h"

#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test1.cpp
 *
 * Basic test of GaussianPdf.
 *
 * This test creates a one dimensional Gaussian with a random mean and
 * variance. It then samples from this distribution and calculates the
 * sample mean and variance for comparison.
 *
 * Results are as follows:
 *
 * @include test1.out
 */

/**
 * Number of samples to take.
 */
unsigned int N = 100000;

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {
  aux::vector mu(1);
  aux::symmetric_matrix sigma(1);
  aux::vector sample;
  double data[N];
  unsigned int i;

  /* set up distribution */
  mu(0) = aux::Random::uniform(-5.0, 5.0);
  sigma(0,0) = aux::Random::uniform(0.0, 5.0);
  aux::GaussianPdf pdf(mu, sigma);

  /* sample from distribution */
  for (i = 0; i < N; i++) {
    sample = pdf.sample();
    data[i] = sample(0);
  }

  /* calculate mean and variance of samples */
  cout << "True mean = " << mu(0) << endl;
  cout << "True variance = " << sigma(0,0) << endl;
  cout << "Sample mean = " << gsl_stats_mean(data, 1, N) << endl;
  cout << "Sample variance = " << gsl_stats_variance(data, 1, N) << endl;
}
