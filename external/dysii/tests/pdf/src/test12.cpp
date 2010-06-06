#include "indii/ml/aux/GaussianMixturePdf.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/Almost2Norm.hpp"
#include "indii/ml/aux/AlmostGaussianKernel.hpp"
#include "indii/ml/aux/Random.hpp"
#include "indii/ml/aux/kde.hpp"

#include <gsl/gsl_statistics_double.h>

#include <iostream>
#include <fstream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test12.cpp
 *
 * Test of indii::ml::aux::selfTreeDensity.
 *
 * This test:
 *
 * @li creates a random multivariate Gaussian mixture,
 * @li samples from this mixture and constructs kernel density
 * approximation,
 * @li calculates the density at the support points of the kernel density
 * approximation, comparing the results of indii::ml::aux::selfTreeDensity
 * and indii::ml::aux::dualTreeDensity.
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
unsigned int P = 5000;

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
  aux::KDTree<> copyTree(tree);
  DiracMixturePdf copyMixtureSamples(mixtureSamples);
  copyTree.setData(&copyMixtureSamples);

  aux::Almost2Norm N;
  aux::AlmostGaussianKernel K(M,H);

  /* density evaluation */
  aux::vector result1(aux::distributedDualTreeDensity(copyTree, tree,
      mixtureSamples.getWeights(), N, K));
  aux::vector result2(aux::distributedSelfTreeDensity(tree,
      mixtureSamples.getWeights(), N, K));

  double err = norm_inf(result1 - result2);
  reduce(world, err, err, boost::mpi::maximum<double>(), 0);

  if (rank == 0) {
    if (err == 0.0) {
      cout << "Passed" << endl;
    } else {
      cout << "Failed" << endl;
      cout << "Max error is " << err << endl;
    }
  }
  
  return 0; 
}

