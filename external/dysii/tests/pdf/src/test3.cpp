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
 * @file test3.cpp
 *
 * Test of DiracMixturePdf.
 *
 * This test creates a GaussianPdf with a random mean and
 * covariance. It then generates uniform samples within 3 standard
 * deviations of the mean and weights them according to the density
 * function of this GaussianPdf, adding them to a DiracMixturePdf
 * distribution. The mean and covariance of the DiracMixturePdf can
 * then be compared with those of the original Gaussian. Furthermore,
 * the DiracMixturePdf is resampled with
 * indii::ml::filter::StratifiedParticleResampler and the new mean
 * and covariance calculated, which should be the same as before the
 * resampling, or very close to.
 *
 * Results are as follows:
 *
 * @include test3.out
 *
 * Note that the covariances for the DiracMixturePdf are expected to be
 * slightly lower than that for the GaussianPdf in general, due to the
 * finite interval over which the samples are taken.
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
  aux::vector mu(M);  // true mean
  aux::symmetric_matrix sigma(M);  // true covariance
  aux::GaussianPdf gaussian(M);
  aux::DiracMixturePdf sampled(M);
  unsigned int i, j;

  if (rank == 0) {
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
  aux::vector sample(mu.size());
  double sd;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      sd = sqrt(var(j));
      sample(j) = mu(j) + aux::Random::uniform(-3.0*sd, 3.0*sd);
    }
    sampled.add(sample, gaussian.calculateDensity(sample));
  }

  mu = sampled.getDistributedExpectation();
  sigma = sampled.getDistributedCovariance();
  if (rank == 0) {
    cout << "True mean" << endl << gaussian.getExpectation() << endl;
    cout << "True covariance" << endl << gaussian.getCovariance() << endl;
    cout << "Sampled mean" << endl << mu << endl;
    cout << "Sampled covariance" << endl << sigma << endl;
  }

  indii::ml::filter::StratifiedParticleResampler resampler;
  sampled = resampler.resample(sampled);
  mu = sampled.getDistributedExpectation();
  sigma = sampled.getDistributedCovariance();

  if (rank == 0) {
    cout << "Resampled mean" << endl << mu << endl;
    cout << "Resampled covariance" << endl << sigma << endl;
  }
}
