//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "DiracPdf.hpp"
//#endif

#include <limits>

using namespace indii::ml::aux;

DiracPdf::DiracPdf() {
  //
}

DiracPdf::DiracPdf(const vector& x) : Pdf(x.size()), vector(x) {
  //
}

DiracPdf::DiracPdf(unsigned int N) : Pdf(N), vector(N) {
  //
}

DiracPdf::~DiracPdf() {
  //
}

void DiracPdf::setDimensions(const unsigned int N, const bool preserve) {
  this->N = N;
  resize(N, preserve);
}

const symmetric_matrix& DiracPdf::getCovariance() const {
  /**
   * @note Not implemented.
   */
}

const symmetric_matrix& DiracPdf::getCovariance() {
  /**
   * @note Not implemented.
   */
}

