#include "indii/ml/aux/GaussianMixturePdf.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/KernelDensityPdf.hpp"
#include "indii/ml/aux/KDTree.hpp"
#include "indii/ml/aux/GaussianKernel.hpp"
#include "indii/ml/aux/PNorm.hpp"
#include "indii/ml/aux/MedianPartitioner.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"
#include "indii/ml/aux/parallel.hpp"
#include "indii/ml/aux/Random.hpp"

#include <gsl/gsl_statistics_double.h>

#include <iostream>
#include <fstream>

using namespace std;

namespace aux = indii::ml::aux;

/**
 * @file test10.cpp
 *
 * Test of KernelDensityPdf and KDTree.
 *
 * This test:
 *
 * @li creates a random multivariate Gaussian mixture,
 * @li samples from this mixture and constructs a \f$kd\f$,
 * @li creates a kernel density estimate from this \f$kd\f$ tree
 * and performs various density calculations using this.
 *
 * Results are as follows:
 *
 * @include test10.out
 *
 * \image html test10_mixture.png "Original Gaussian mixture"
 * \image latex test10_mixture.eps "Original Gaussian mixture"
 * \image html test10_tree.png "Kernel density approximation"
 * \image latex test10_tree.eps "Kernel density approximation"
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
 * Scaling parameter.
 */
double H = 0.25 * std::pow((double)4/(P*(M+2)), (double)1/(M+4));

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
  //int seed = static_cast<int>(aux::Random::uniform(0, 1000000));
  //std::cerr << "seed = " << seed << std::endl;
  //aux::Random::seed(seed);
  aux::Random::seed(781634);

  unsigned int i, j;

  /* create distribution */
  aux::GaussianMixturePdf mixture(M);
  for (i = 0; i < COMPONENTS; i++) {
    mixture.add(createRandomGaussian(M),
        aux::Random::uniform(0.5,1.0));
  }

  /* sample from distribution */
  aux::DiracMixturePdf mixtureSamples(mixture, P);

  /* construct KD tree */
  aux::KDTree<aux::MedianPartitioner> tree(&mixtureSamples);
  aux::PNorm<2> N;
  aux::GaussianKernel K(M, H);
  aux::KernelDensityPdf<aux::PNorm<2>,aux::GaussianKernel> kd(&tree, N, K);

  /* sample from kernel density */
  aux::DiracMixturePdf kdSamples(kd, P);

  /* importance sample from kernel density */
  aux::GaussianPdf importance(mixture.getExpectation(),
      mixture.getCovariance());
  aux::DiracMixturePdf kdImportanceSamples(M);
  double kdDensity, importanceDensity;
  aux::vector sample(M);

  for (i = 0; i < P; i++) {
    sample = importance.sample();
    importanceDensity = importance.densityAt(sample);
    kdDensity = kd.densityAt(sample);
    kdImportanceSamples.add(sample, kdDensity/importanceDensity);
  }

  cout << "Mixture mean" << endl <<
      mixture.getExpectation() << endl;
  cout << "Mixture covariance" << endl <<
      mixture.getCovariance() << endl;
  cout << "Sample mean" << endl <<
      mixtureSamples.getExpectation() << endl;
  cout << "Sample covariance" << endl <<
      mixtureSamples.getCovariance() << endl;
  cout << "Kernel density mean" << endl <<
      kd.getExpectation() << endl;
  cout << "Kernel density tree covariance" << endl <<
      kd.getCovariance() << endl;
  cout << "Kernel density sample mean" << endl <<
      kdSamples.getExpectation() << endl;
  cout << "Kernel density sample covariance" << endl <<
      kdSamples.getCovariance() << endl;
  cout << "Kernel density importance sample mean" << endl <<
      kdImportanceSamples.getExpectation() << endl;
  cout << "Kernel density importance sample covariance" << endl <<
      kdImportanceSamples.getCovariance() << endl;
      
  /* calculate bounds */
  KDTreeNode* kdRoot = dynamic_cast<KDTreeNode*>(tree.getRoot());
  const aux::vector& lower = *kdRoot->getLower();
  const aux::vector& upper = *kdRoot->getUpper();

  /* output for plots */
  ofstream fMixture("results/test10_mixture.out");
  ofstream fKD("results/test10_tree.out");
  aux::vector coord(M);
  DiracMixturePdf query(M);
  double x, y, density;
  
  for (i = 0; i < RES; i++) {
    x = lower(0) + (upper(0) - lower(0)) * i / RES;
    coord(0) = x;
    for (j = 0; j < RES; j++) {
      y = lower(1) + (upper(1) - lower(1)) * j / RES;
      coord(1) = y;
      
      density = mixture.densityAt(coord);
      fMixture << x << '\t' << y << '\t' << density << endl;
      
      //density = kd.densityAt(coord);
      //fKD << x << '\t' << y << '\t' << density << endl;
      query.add(coord);
    }
    
    /* end isolines */
    fMixture << endl;
    //fKD << endl;
  }
 
  /* dual tree query */
  aux::KDTree<aux::MedianPartitioner> queryTree(&query);
  aux::vector queryDensity(kd.densityAt(queryTree));
  for (i = 0; i < RES; i++) {
    for (j = 0; j < RES; j++) {
      noalias(coord) = query.get(i*RES+j);
      x = coord(0);
      y = coord(1);
      fKD << x << '\t' << y << '\t' << queryDensity(i*RES+j) << endl;
    }
    fKD << endl;
  }
 
  return 0; 
}

