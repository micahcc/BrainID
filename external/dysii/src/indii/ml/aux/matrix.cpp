//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "matrix.hpp"
  #include "vector.hpp"
//#endif

#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

namespace lapack = boost::numeric::bindings::lapack;

using namespace indii::ml::aux;

void indii::ml::aux::inv(matrix& A, matrix& AI) {
  /* pre-condition */
  assert (A.size1() == A.size2());
  assert (AI.size1() == A.size1());
  assert (AI.size2() == A.size2());

  const unsigned int N = A.size1();
  const identity_matrix I(N);

  AI = I;
  #ifndef NDEBUG
  int ierr = lapack::gesv(A, AI);
  assert (ierr == 0);
  #else
  lapack::gesv(A, AI);
  #endif
}

