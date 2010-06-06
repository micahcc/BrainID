#include "indii/ml/aux/WienerProcess.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/Random.hpp"

#include <iostream>
#include <fstream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test4.cpp
 *
 * Test of WienerProcess.
 *
 * This test creates a univariate WienerProcess and simulates several,
 * plotting:
 *
 * @li the actual mean and variance at each point
 * @li the expected mean and variance at each point.
 * @li the mean and variance at each point calculated using importance
 * sampling, so as to test the densityAt() function.
 *
 * Results are as follows:
 *
 * \image html test4.png "Results"
 * \image latex test4.eps "Results"
 */

/**
 * Dimensionality of the process.
 */
const unsigned int M = 1;

/**
 * Number of sample trajectories.
 */
const unsigned int N = 1000;

/**
 * Length of each trajectory.
 */
const unsigned int LENGTH = 500;

/**
 * Length of time step.
 */
const unsigned int DELTA = 3;

/**
 * Number of importance samples.
 */
const unsigned int P = 10000;

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {
  /* set up distributions */
  unsigned int i, j;

  std::ofstream fact("results/test4_actual.out"); // actual statistics
  std::ofstream fexp("results/test4_expected.out"); // expected statistics
  std::ofstream fimp("results/test4_sample.out"); // importance sampling

  aux::WienerProcess<unsigned int> wiener(1);
  std::vector<aux::DiracMixturePdf> ts; // for calculating mean and variance
  aux::DiracPdf trajectory(M);

  for (i = 0; i < LENGTH; i++) {
    ts.push_back(aux::DiracMixturePdf(M));
  }

  /* calculate trajectories */
  for (i = 0; i < N; i++) {
    trajectory(0) = 0.0;
    ts[0].add(trajectory);

    for (j = 1; j < LENGTH; j+=DELTA) {
      trajectory += wiener.sample(DELTA);

      ts[j/DELTA].add(trajectory);
    }
  }

  /* output mean and std. dev. of trajectories at each time point,
     importance sample to achieve same result */
  for (j = 0; j < LENGTH; j+=DELTA) {
    fact << j << '\t';
    fact << ts[j/DELTA].getExpectation()(0) << '\t';
    fact << ts[j/DELTA].getCovariance()(0,0) << endl;
  }

  /* output expected mean and covariance at each time point */
  for (j = 0; j < LENGTH; j+=DELTA) {
    fexp << j << '\t';
    fexp << 0.0 << '\t';
    fexp << j << endl;
  }

  /* importance sample mean and std.dev of trajectories at each time
     point */
  aux::GaussianPdf gaussian(M);
  aux::GaussianPdf q(M); // proposal distribution
  aux::DiracMixturePdf sampled(M);
  aux::vector s(M);
  double w;

  q.setExpectation(aux::zero_vector(M));
  q.setCovariance(5000.0*aux::identity_matrix(M));

  for (j = 0; j < LENGTH; j+=DELTA) {
    gaussian.setExpectation(ts[j/DELTA].getExpectation());
    gaussian.setCovariance(ts[j/DELTA].getCovariance());

    sampled.clear();
    for (i = 0; i < P; i++) {
      s = q.sample();
      w = gaussian.densityAt(s) / q.densityAt(s);
      sampled.add(s, w);
    }

    fimp << j << '\t';
    fimp << sampled.getExpectation()(0) << '\t';
    fimp << sampled.getCovariance()(0,0) << endl;
  }

}
