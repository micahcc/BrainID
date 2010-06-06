#include "AlmostGaussianKernel.hpp"

using namespace indii::ml::aux;

AlmostGaussianKernel::AlmostGaussianKernel() : Kernel() {
  //
}

AlmostGaussianKernel::AlmostGaussianKernel(const unsigned int N,
    const double h) : Kernel(h) {
  ZI = 1.0 / pow(h * sqrt(2.0*M_PI), N);
  E = -1.0 / (2.0 * pow(h,2));
}

AlmostGaussianKernel::~AlmostGaussianKernel() {
  //
}

