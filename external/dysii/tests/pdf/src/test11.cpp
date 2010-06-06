#include "indii/ml/aux/GaussianMixturePdf.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/KernelDensityMixturePdf.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/Random.hpp"
#include "indii/ml/aux/kde.hpp"

#include <gsl/gsl_statistics_double.h>

#include <iostream>
#include <fstream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test11.cpp
 *
 * Test of KernelDensityMixturePdf.
 *
 * This test:
 *
 * @li creates a random multivariate Gaussian mixture,
 * @li samples from this mixture and constructs a \f$kd\f$ tree,
 * @li Constructs a KernelDensityPdf approximation of the original Gaussian
 * mixture from this \f$kd\f$ tree.
 *
 * Results are as follows:
 *
 * @include test11.out
 */

/**
 * Dimensionality of the distribution.
 */
unsigned int M = 2;

/**
 * Number of components in the Gaussian mixture.
 */
unsigned int COMPONENTS = 4;

/**
 * Number of samples to take.
 */
unsigned int P = 1000;

/**
 * Resolution of plots.
 */
unsigned int RES = 200;

/**
 * Bandwidth.
 */
double H = 0.25 * aux::hopt(M,P);

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
    const double minMean = -1.0, const double maxMean = 1.0,
    const double minCov = -1.0, const double maxCov = 1.0) {
  aux::vector mu(M);
  aux::symmetric_matrix sigma(M);
  aux::lower_triangular_matrix L(M,M);

  unsigned int i, j;

  /* mean */
  for (i = 0; i < M; i++) {
    mu(i) = aux::Random::uniform(minMean, maxMean);
  }

  /* covariance */
  for (i = 0; i < M; i++) {
    for (j = 0; j <= i; j++) {
      L(i,j) = aux::Random::gaussian((maxCov + minCov) / 2.0,
          (maxCov - minCov) / 2.0);
    }
  }
  sigma = prod(L, trans(L)); // ensures cholesky decomposable

  return aux::GaussianPdf(mu, sigma);
}

/**
 * Run tests.
 */
int main(int argc, char* argv[]) {
  /* mpi */
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  unsigned int rank = world.rank();  
  unsigned int size = world.size();

  //int seed = static_cast<int>(aux::Random::uniform(0, 1000000));
  //std::cerr << "seed = " << seed << std::endl;
  //aux::Random::seed(seed);
  //aux::Random::seed(781634 + rank);

  unsigned int i;

  /* create distribution and broadcast */
  aux::GaussianMixturePdf mixture(M);
  if (rank == 0) {
    for (i = 0; i < COMPONENTS; i++) {
      mixture.add(createRandomGaussian(M),
          aux::Random::uniform(0.5,1.0));
    }
  }
  boost::mpi::broadcast(world, mixture, 0);

  /* sample from distribution */
  aux::DiracMixturePdf mixtureSamples(mixture, P / size);
  mixtureSamples.redistributeBySpace();

  /* construct kd tree */
  aux::KDTree<> tree(&mixtureSamples);
  aux::Almost2Norm N;
  aux::AlmostGaussianKernel K(M,H);
  aux::KernelDensityPdf<> kd(&tree, N, K);
  aux::KernelDensityMixturePdf<> kdMixture(kd,
      mixtureSamples.getTotalWeight());

  /* sample from kernel density mixture */
  std::vector<aux::vector> xs = kdMixture.distributedSample(P);
  aux::DiracMixturePdf kdSamples(M);
  for (i = 0; i < xs.size(); i++) {
    kdSamples.add(xs[i]);
  }

  /* importance sample from kernel density */
  aux::GaussianPdf importance(mixture.getExpectation(),
      mixture.getCovariance());
  aux::DiracMixturePdf kdImportanceSamples(M), querySamples(M);
  
  for (i = 0; i < P / size; i++) {
    querySamples.add(importance.sample());
  }
  querySamples.redistributeBySpace();

  aux::KDTree<aux::MedianPartitioner> queryTree(&querySamples);
  aux::vector kdDensities(kdMixture.distributedDensityAt(queryTree));
  //noalias(kdDensities) = kdMixture.distributedDensityAt(samples);

  for (i = 0; i < querySamples.getSize(); i++) {
    kdImportanceSamples.add(querySamples.get(i), kdDensities(i) /
        importance.densityAt(querySamples.get(i)));
  }

  if (rank == 0) {
    cout << "Mixture mean" << endl <<
        mixture.getExpectation() << endl;
    cout << "Mixture covariance" << endl <<
        mixture.getCovariance() << endl;
    cout << "Sample mean" << endl <<
        mixtureSamples.getDistributedExpectation() << endl;
    cout << "Sample covariance" << endl <<
        mixtureSamples.getDistributedCovariance() << endl;
    cout << "Kernel density mixture mean" << endl <<
        kdMixture.getDistributedExpectation() << endl;
    cout << "Kernel density mixture covariance" << endl <<
        kdMixture.getDistributedCovariance() << endl;
    cout << "Kernel density mixture sample mean" << endl <<
        kdSamples.getDistributedExpectation() << endl;
    cout << "Kernel density mixture sample covariance" << endl <<
        kdSamples.getDistributedCovariance() << endl;
    cout << "Kernel density mixture importance sample mean" << endl <<
        kdImportanceSamples.getDistributedExpectation() << endl;
    cout << "Kernel density mixture importance sample covariance" << endl <<
        kdImportanceSamples.getDistributedCovariance() << endl;
  } else {
    mixtureSamples.getDistributedExpectation();
    mixtureSamples.getDistributedCovariance();
    kdMixture.getDistributedExpectation();
    kdMixture.getDistributedCovariance();
    kdSamples.getDistributedExpectation();
    kdSamples.getDistributedCovariance();
    kdImportanceSamples.getDistributedExpectation();
    kdImportanceSamples.getDistributedCovariance();
  }
  
  return 0; 
}

