#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/Random.hpp"
#include "indii/ml/filter/StratifiedParticleResampler.hpp"

#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test8.cpp
 *
 * Test of DiracMixturePdf with distributed storage.
 *
 * As test3.cpp, but with use of distributed storage.
 *
 * Results are as follows:
 *
 * @include test8.out
 */

/**
 * Dimensionality of the Gaussian.
 */
unsigned int M = 4;

/**
 * Number of samples to take.
 */
unsigned int N = 10000;

/**
 * Run tests.
 */
int main(int argc, char* argv[]) {
  /* mpi */
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  int rank = world.rank();

  /* set up distributions */
  aux::GaussianPdf gaussian(M);
  aux::DiracMixturePdf sampled(M);
  unsigned int i, j;

  if (rank == 0) {
    aux::vector mu(M);  // true mean
    aux::symmetric_matrix sigma(M);  // true covariance
    aux::lower_triangular_matrix tmp(M,M);

    for (i = 0; i < M; i++) {
      mu(i) = aux::Random::uniform(-5.0, 10.0);
    }
    
    for (i = 0; i < M; i++) {
      for (j = 0; j <= i; j++) {
        tmp(i,j) = aux::Random::uniform(0.0, 5.0);
      }
    }
    noalias(sigma) = prod(tmp, trans(tmp)); // ensures cholesky decomposition

    gaussian.setExpectation(mu);
    gaussian.setCovariance(sigma);
  }
  boost::mpi::broadcast(world, gaussian, 0);

  /* uniformly sample from within 3 standard deviations of mean and
     add to sampled distribution, weighted by Gaussian probability
     density */
  aux::vector var = aux::diag(gaussian.getCovariance());
  aux::vector sample(M);
  double sd;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      sd = sqrt(var(j));
      sample(j) = gaussian.getExpectation()(j) +
	  aux::Random::uniform(-3.0*sd, 3.0*sd);
    }
    sampled.add(sample, gaussian.calculateDensity(sample));
  }

  indii::ml::filter::StratifiedParticleResampler resampler;
  if (rank == 0) {
    cout << "True mean" << endl << gaussian.getExpectation() << endl;
    cout << "True covariance" << endl << gaussian.getCovariance() << endl;

    cout << "Sampled mean" << endl <<
        sampled.getDistributedExpectation() << endl;
    cout << "Sampled covariance" << endl <<
        sampled.getDistributedCovariance() << endl;

    sampled = resampler.resample(sampled);
    cout << "Resampled mean" << endl <<
        sampled.getDistributedExpectation() << endl;
    cout << "Resampled covariance" << endl <<
        sampled.getDistributedCovariance() << endl;
  } else {
    sampled.getDistributedExpectation();
    sampled.getDistributedCovariance();
    sampled = resampler.resample(sampled);
    sampled.getDistributedExpectation();
    sampled.getDistributedCovariance();
  }

  return 0;
}
