#include "indii/ml/aux/GaussianMixturePdf.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/parallel.hpp"
#include "indii/ml/aux/Random.hpp"

#include "boost/mpi/environment.hpp"
#include "boost/mpi/communicator.hpp"

#include <gsl/gsl_statistics_double.h>

#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test7.cpp
 *
 * Test of GaussianMixturePdf with distributed storage.
 *
 * This test creates a random multivariate Gaussian mixture with
 * components distributed across several nodes. It goes on to test the
 * various distributed methods to ensure invariance of the
 * distribution under these manipulations.
 *
 * Results follow.
 *
 * @include test7.out
 */

/**
 * Dimensionality of the Gaussian mixture.
 */
unsigned int M = 10;

/**
 * Number of components in the Gaussian mixture.
 */
unsigned int COMPONENTS = 50;

/**
 * Number of samples to take.
 */
unsigned int N = 10000;

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
 * Determines whether a vector is the zero vector.
 */
bool isZero(const aux::vector& x) {
  unsigned int i;
  for (i = 0; i < x.size(); i++) {
    if (x(i) != 0.0) {
      return false;
    }
  }
  return true;
}

/**
 * Determine whether a matrix is the zero matrix.
 */
bool isZero(const aux::matrix& A) {
  unsigned int i, j;
  for (i = 0; i < A.size1(); i++) {
    for (j = 0; j < A.size2(); j++) {
      if (A(i,j) != 0.0) {
	return false;
      }
    }
  }
  return true;
}

/**
 * Run tests.
 */
int main(int argc, char* argv[]) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  const int rank = world.rank();
  
  unsigned int i;
  bool passed;

  aux::vector mu(M);  // true mean
  aux::symmetric_matrix sigma(M);  // true covariance
  aux::vector dmu(M);  // distributed mean
  aux::symmetric_matrix dsigma(M);  // distributed covariance
  aux::vector rdmu(M);  // rotated distributed mean
  aux::symmetric_matrix rdsigma(M);  // rotated distributed covariance
  aux::vector nmu(M);  // normalised distributed mean
  aux::symmetric_matrix nsigma(M);  // normalised distributed covariance

  /* create local distribution */
  aux::GaussianMixturePdf pdf(M);
  if (rank == 0) {
    for (i = 0; i < COMPONENTS; i++) {
      pdf.add(createRandomGaussian(M), aux::Random::uniform(0.0,1.0));
    }
    noalias(mu) = pdf.getExpectation();
    noalias(sigma) = pdf.getCovariance();
  }
  world.barrier();
  cout << rank << ": " << pdf.getSize() << " components, " <<
      pdf.getTotalWeight() << " weight" << endl;

  /* distribute across nodes */
  if (rank == 0) {
    cout << endl << "redistributeBySize() ";
  }
  world.barrier();
  pdf.redistributeBySize();

  noalias(dmu) = pdf.getDistributedExpectation();
  noalias(dsigma) = pdf.getDistributedCovariance();

  if (rank == 0) {
    passed = true;
    passed &= isZero(mu - dmu);
    passed &= isZero(sigma - dsigma);
    if (passed) {
      cout << "passed";
    } else {
      cout << "failed, difference is:" << endl;
      cout << mu - dmu << endl;
      cout << sigma - dsigma << endl;
    }
    cout << endl;
  }
  world.barrier();
  cout << rank << ": " << pdf.getSize() << " components, " <<
      pdf.getTotalWeight() << " weight" << endl;

  /* redistribute across nodes */
  if (rank == 0) {
    cout << endl << "redistributeByWeight() ";
  }
  world.barrier();
  pdf.redistributeByWeight();

  noalias(dmu) = pdf.getDistributedExpectation();
  noalias(dsigma) = pdf.getDistributedCovariance();

  if (rank == 0) {
    passed = true;
    passed &= isZero(mu - dmu);
    passed &= isZero(sigma - dsigma);
    if (passed) {
      cout << "passed";
    } else {
      cout << "failed, difference is:" << endl;
      cout << mu - dmu << endl;
      cout << sigma - dsigma << endl;
    }
    cout << endl;
  }
  world.barrier();
  cout << rank << ": " << pdf.getSize() << " components, " <<
      pdf.getTotalWeight() << " weight" << endl;

  /* rotate */
  if (rank == 0) {
    cout << endl << "rotate() ";
  }
  world.barrier();
  aux::rotate(pdf);

  noalias(rdmu) = pdf.getDistributedExpectation();
  noalias(rdsigma) = pdf.getDistributedCovariance();

  if (rank == 0) {
    passed = true;
    passed &= isZero(dmu - rdmu);
    passed &= isZero(dsigma - rdsigma);
    if (passed) {
      cout << "passed";
    } else {
      cout << "failed, difference is:" << endl;
      cout << dmu - rdmu << endl;
      cout << dsigma - rdsigma << endl;
    }
    cout << endl;
  }
  world.barrier();
  cout << rank << ": " << pdf.getSize() << " components, " <<
      pdf.getTotalWeight() << " weight" << endl;

  /* normalise */
  if (rank == 0) {
    cout << endl << "distributedNormalise() ";
  }
  world.barrier();
  pdf.distributedNormalise();

  noalias(nmu) = pdf.getDistributedExpectation();
  noalias(nsigma) = pdf.getDistributedCovariance();

  if (rank == 0) {
    passed = true;
    passed &= isZero(rdmu - nmu);
    passed &= isZero(rdsigma - nsigma);
    if (passed) {
      cout << "passed";
    } else {
      cout << "failed, difference is:" << endl;
      cout << rdmu - nmu << endl;
      cout << rdsigma - nsigma << endl;
    }
    cout << endl;
  }
  world.barrier();
  cout << rank << ": " << pdf.getSize() << " components, " <<
      pdf.getTotalWeight() << " weight" << endl;
}

