#include "FixableStateModel.hpp"

namespace aux = indii::ml::aux;
namespace ublas = boost::numeric::ublas;

using namespace indii::ml::filter;

FixableStateModel::FixableStateModel() {
  //
}

FixableStateModel::FixableStateModel(const unsigned int N) : N(N), F(0),
    fixed(N), projectCondense(N,N) {
  unsigned int i;
  
  fixed.clear();
  projectCondense.clear();
  for (i = 0; i < N; i++) {
    projectCondense(i,i) = 1;
  }
}

FixableStateModel::~FixableStateModel() {
  //
}

unsigned int FixableStateModel::getVariableSize() const {
  return N;
}

unsigned int FixableStateModel::getFixedSize() const {
  return F;
}

void FixableStateModel::fix(const unsigned int i, const double value) {
  /* pre-condition */
  assert (i < N + F);

  unsigned int row, col;
  bool isFixed = true; // is variable already fixed?
  
  for (row = 0; row < projectCondense.size1(); row++) {
    if (projectCondense(row,i) == 1) {
      isFixed = false;
      break;
    }
  }

  if (!isFixed) {
    /* update projection matrix */
    /**
     * @todo Preservation in resize() is not yet implemented in uBLAS
     * for sparse matrices. Copy into dense matrix at present,
     * sparse matrices are used here for computational rather than
     * storage efficiency after all. Review in future if
     * preservation for sparse matrices is implemented in uBLAS.
     */
    aux::matrix projectCondenseDense(projectCondense);
    ublas::range cols(0, N + F);
    ublas::range to(row, N - 1);
    ublas::range from(row + 1, N);
    ublas::matrix_range<aux::matrix>(projectCondenseDense, to, cols) =
        ublas::matrix_range<aux::matrix>(projectCondenseDense, from, cols);
    projectCondenseDense.resize(N - 1, N + F, true);
    projectCondense.resize(N - 1, N + F, false);
    projectCondense.clear();
    for (col = 0; col < projectCondenseDense.size2(); col++) {
      for (row = 0; row < projectCondenseDense.size1(); row++) {
        if (projectCondenseDense(row,col) == 1) {
          projectCondense(row,col) = 1;
    	}
      }
    }
    
    N--;
    F++;
  }
  
  /* update fixed variables */
  fixed(i) = value;

  /* post-condition */
  assert (projectCondense.size1() == N);
  assert (projectCondense.size2() == N + F);
}

