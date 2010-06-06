#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/DiracPdf.hpp"
#include "indii/ml/aux/UniformPdf.hpp"
#include "indii/ml/aux/KernelDensityPdf.hpp"
#include "indii/ml/aux/GaussianMixturePdf.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/KDTree.hpp"

#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"

#include <fstream>
#include <iostream>
#include <ios>

namespace aux = indii::ml::aux;

/**
 * @file test5.cpp
 *
 * Test of serialization, input and output, for all pdfs. A random pdf
 * is constructed for each type, serialized and output to a file. A
 * new pdf is constructed from the serialization in the file and
 * compared to the original for equality.
 *
 * Results are as follows:
 *
 * @include test5.out
 */

/**
 * Dimensionality of the distributions.
 */
unsigned int N = 4;

/**
 * Number of components for Gaussian mixture test.
 */
unsigned int GAUSSIAN_MIXTURE_COMPONENTS = 12;

/**
 * Number of components for Dirac mixture test.
 */
unsigned int DIRAC_MIXTURE_COMPONENTS = 1000;

/**
 * Create random Dirac distribution.
 *
 * @param M Dimensionality of the Dirac.
 * @param minMean Minimum value of any component of the mean.
 * @param maxMean Maximum value of any component of the mean.
 *
 * Dirac with given dimensionality, with mean randomly generated
 * uniformly from within the given bounds.
 */
aux::DiracPdf createRandomDirac(const unsigned int M,
    const double minMean = -5.0, const double maxMean = 5.0) {
  aux::DiracPdf dirac(M);
  unsigned int i;

  /* mean */
  for (i = 0; i < M; i++) {
    dirac(i) = aux::Random::uniform(minMean, maxMean);
  }

  return dirac;
}

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
 * Create random uniform distribution.
 *
 * @param M Dimensionality of the distribution.
 * @param minBound Minimum value of any component of a bound.
 * @param maxBound Maximum value of any component of a bound.
 *
 * @return Uniform distribution with given dimensionality, with lower and
 * upper bound randomly generated uniformly from within the given bounds.
 */
aux::UniformPdf createRandomUniform(const unsigned int M,
    const double minBound = -5.0, const double maxBound = 5.0) {
  aux::vector lower(M), upper(M);
  double a, b;
  unsigned int i;
  
  for (i = 0; i < M; i++) {
    a = aux::Random::uniform(minBound, maxBound);
    b = aux::Random::uniform(minBound, maxBound);
    
    if (a > b) {
      upper(i) = a;
      lower(i) = b;
    } else {
      upper(i) = b;
      lower(i) = a;
    }
  }
  
  return aux::UniformPdf(lower, upper);
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
 * Test of GaussianPdf.
 */
bool testGaussianPdf() {
  bool passed;
  aux::vector mu(N);
  aux::symmetric_matrix sigma(N);

  aux::GaussianPdf gaussianIn;
  const aux::GaussianPdf gaussianOut(createRandomGaussian(N));
  {
    std::ofstream outFile("data/GaussianPdf.obj", std::ios::binary);
    boost::archive::binary_oarchive outArchive(outFile);
    outArchive << gaussianOut;
  }
  {
    std::ifstream inFile("data/GaussianPdf.obj", std::ios::binary);
    boost::archive::binary_iarchive inArchive(inFile);
    inArchive >> gaussianIn;
  }
  
  passed = isZero(gaussianOut.getExpectation() - gaussianIn.getExpectation());
  passed &= isZero(gaussianOut.getCovariance() - gaussianIn.getCovariance());

  return passed;
}

/**
 * Test of DiracPdf.
 */
bool testDiracPdf() {
  bool passed;

  aux::DiracPdf diracIn;
  const aux::DiracPdf diracOut(createRandomDirac(N));
  {
    std::ofstream outFile("data/DiracPdf.obj", std::ios::binary);
    boost::archive::binary_oarchive outArchive(outFile);
    outArchive << diracOut;
  }
  {
    std::ifstream inFile("data/DiracPdf.obj", std::ios::binary);
    boost::archive::binary_iarchive inArchive(inFile);
    inArchive >> diracIn;
  }
  
  passed = isZero(diracOut.getExpectation() - diracIn.getExpectation());

  return passed;
}

/**
 * Test of UniformPdf.
 */
bool testUniformPdf() {
  bool passed;

  aux::UniformPdf uniformIn;
  aux::UniformPdf uniformOut(createRandomUniform(N));
  {
    std::ofstream outFile("data/UniformPdf.obj", std::ios::binary);
    boost::archive::binary_oarchive outArchive(outFile);
    const aux::UniformPdf tmp(uniformOut);
    outArchive << tmp;
  }
  {
    std::ifstream inFile("data/UniformPdf.obj", std::ios::binary);
    boost::archive::binary_iarchive inArchive(inFile);
    inArchive >> uniformIn;
  }
  
  passed = isZero(uniformOut.getExpectation() - uniformIn.getExpectation());
  passed &= isZero(uniformOut.getCovariance() - uniformIn.getCovariance());

  return passed;
}

/**
 * Test of GaussianMixturePdf.
 */
bool testGaussianMixturePdf() {
  bool passed;
  unsigned int i;
  aux::vector mu(N);
  aux::symmetric_matrix sigma(N);

  aux::GaussianMixturePdf gaussianMixIn;
  aux::GaussianMixturePdf gaussianMixOut(N);
  for (i = 0; i < GAUSSIAN_MIXTURE_COMPONENTS; i++) {
    gaussianMixOut.add(createRandomGaussian(N),
        aux::Random::uniform(0.0,1.0));
  }

  {
    std::ofstream outFile("data/GaussianMixturePdf.obj", std::ios::binary);
    boost::archive::binary_oarchive outArchive(outFile);
    const aux::GaussianMixturePdf tmp(gaussianMixOut);
    outArchive << tmp;
  }
  {
    std::ifstream inFile("data/GaussianMixturePdf.obj", std::ios::binary);
    boost::archive::binary_iarchive inArchive(inFile);
    inArchive >> gaussianMixIn;
  }
  
  passed = gaussianMixIn.getSize() == gaussianMixOut.getSize();
  for (i = 0; i < gaussianMixIn.getSize(); i++) {
    passed &= gaussianMixIn.getWeight(i) == gaussianMixOut.getWeight(i);
    passed &= isZero(gaussianMixIn.get(i).getExpectation() -
        gaussianMixOut.get(i).getExpectation());
    passed &= isZero(gaussianMixIn.get(i).getCovariance() -
        gaussianMixOut.get(i).getCovariance());
  }

  return passed;
}

/**
 * Test of DiracMixturePdf.
 */
bool testDiracMixturePdf() {
  bool passed;
  unsigned int i;
  aux::vector mu(N);

  aux::DiracMixturePdf diracMixIn;
  aux::DiracMixturePdf diracMixOut(N);
  for (i = 0; i < DIRAC_MIXTURE_COMPONENTS; i++) {
    diracMixOut.add(createRandomDirac(N),
        aux::Random::uniform(0.0,1.0));
  }

  {
    std::ofstream outFile("data/DiracMixturePdf.obj", std::ios::binary);
    boost::archive::binary_oarchive outArchive(outFile);
    const aux::DiracMixturePdf tmp(diracMixOut);
    outArchive << tmp;
  }
  {
    std::ifstream inFile("data/DiracMixturePdf.obj", std::ios::binary);
    boost::archive::binary_iarchive inArchive(inFile);
    inArchive >> diracMixIn;
  }
  
  passed = diracMixIn.getSize() == diracMixOut.getSize();
  for (i = 0; i < diracMixIn.getSize(); i++) {
    passed &= diracMixIn.getWeight(i) == diracMixOut.getWeight(i);
    passed &= isZero(diracMixIn.get(i).getExpectation() -
        diracMixOut.get(i).getExpectation());
  }

  return passed;
}

/**
 * Test of KernelDensityPdf.
 */
bool testKernelDensityPdf() {
  bool passed;
  unsigned int i;

  aux::DiracMixturePdf diracMixOut(N);
  for (i = 0; i < DIRAC_MIXTURE_COMPONENTS; i++) {
    diracMixOut.add(createRandomDirac(N),
        aux::Random::uniform(0.0,1.0));
  }

  aux::Almost2Norm norm;
  aux::AlmostGaussianKernel kernel(N, 1.0);
  aux::KDTree<> treeOut(&diracMixOut);
  aux::KernelDensityPdf<> kdOut(&treeOut, norm, kernel);
  aux::KernelDensityPdf<> kdIn;

  {
    std::ofstream outFile("data/KernelDensityMixturePdf.obj",
        std::ios::binary);
    boost::archive::binary_oarchive outArchive(outFile);
    const aux::KernelDensityPdf<> tmp(kdOut);
    outArchive << tmp;
  }
  {
    std::ifstream inFile("data/KernelDensityMixturePdf.obj",
        std::ios::binary);
    boost::archive::binary_iarchive inArchive(inFile);
    inArchive >> kdIn;
  }
  
  passed = true;
  passed &= isZero(kdIn.getExpectation() - kdOut.getExpectation());
  passed &= isZero(kdIn.getCovariance() - kdOut.getCovariance());

  return passed;
}

/**
 * Run tests.
 */
int main(int argc, const char* argv[]) {

  std::cout << "GaussianPdf ";
  if (testGaussianPdf()) {
    std::cout << "passed";
  } else {
    std::cout << "failed";
  }
  std::cout << std::endl;

  std::cout << "DiracPdf ";
  if (testDiracPdf()) {
    std::cout << "passed";
  } else {
    std::cout << "failed";
  }
  std::cout << std::endl;

  std::cout << "UniformPdf ";
  if (testUniformPdf()) {
    std::cout << "passed";
  } else {
    std::cout << "failed";
  }
  std::cout << std::endl;

  //std::cout << "KernelDensityPdf ";
  //if (testKernelDensityPdf()) {
  //  std::cout << "passed";
  //} else {
  //  std::cout << "failed";
  //}
  //std::cout << std::endl;

  std::cout << "GaussianMixturePdf ";
  if (testGaussianMixturePdf()) {
    std::cout << "passed";
  } else {
    std::cout << "failed";
  }
  std::cout << std::endl;

  std::cout << "DiracMixturePdf ";
  if (testDiracMixturePdf()) {
    std::cout << "passed";
  } else {
    std::cout << "failed";
  }
  std::cout << std::endl;

  return 0;
}
