#ifndef INDII_ML_AUX_MATRIX_HPP
#define INDII_ML_AUX_MATRIX_HPP

#include "vector.hpp"

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_sparse.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/symmetric.hpp"
#include "boost/numeric/ublas/triangular.hpp"
#include "boost/numeric/ublas/banded.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/storage.hpp"

namespace indii {
  namespace ml {
    namespace aux {

      /**
       * General matrix.
       */
      typedef boost::numeric::ublas::matrix<double,
          boost::numeric::ublas::column_major,
          boost::numeric::ublas::unbounded_array<double> > matrix;

      /**
       * Symmetric matrix.
       */
      typedef boost::numeric::ublas::symmetric_matrix<double,
          boost::numeric::ublas::lower, boost::numeric::ublas::column_major,
          boost::numeric::ublas::unbounded_array<double> > symmetric_matrix;

      /**
       * Lower triangular matrix.
       */
      typedef boost::numeric::ublas::triangular_matrix<double,
          boost::numeric::ublas::lower, boost::numeric::ublas::column_major,
          boost::numeric::ublas::unbounded_array<double> >
          lower_triangular_matrix;

      /**
       * Upper triangular matrix.
       */
      typedef boost::numeric::ublas::triangular_matrix<double,
          boost::numeric::ublas::upper, boost::numeric::ublas::column_major,
          boost::numeric::ublas::unbounded_array<double> >
          upper_triangular_matrix;

      /**
       * Identity matrix.
       */
      typedef boost::numeric::ublas::identity_matrix<double> identity_matrix;
      
      /**
       * Zero matrix.
       */
      typedef boost::numeric::ublas::zero_matrix<double> zero_matrix;

      /**
       * Scalar matrix.
       */
      typedef boost::numeric::ublas::scalar_matrix<double> scalar_matrix;

      /**
       * Banded matrix.
       */
      typedef boost::numeric::ublas::banded_matrix<double,
          boost::numeric::ublas::column_major,
          boost::numeric::ublas::unbounded_array<double> > banded_matrix;

      /**
       * Sparse matrix.
       */
      typedef boost::numeric::ublas::mapped_matrix<double,
          boost::numeric::ublas::column_major> sparse_matrix;
      //typedef boost::numeric::ublas::compressed_matrix<double,
      //    boost::numeric::ublas::column_major> sparse_matrix;
      // ^ seems to cause segfaults in some situations, perhaps when
      //   assertions are disabled?

      /**
       * Projection matrix.
       */
      typedef boost::numeric::ublas::mapped_matrix<short int,
          boost::numeric::ublas::column_major> projection_matrix;

      /**
       * Inverse of a square matrix.
       *
       * @param A matrix to invert.
       * @param AI matrix into which to write the inverse of A.
       *
       * @warning Will change the contents of @p A!
       */
      void inv(matrix& A, matrix& AI);

      /**
       * Diagonal of a square matrix.
       *
       * @param A square matrix.
       */
      template <class T>
      boost::numeric::ublas::matrix_vector_range<T> diag(T& A);

      /**
       * Convert vector to matrix.
       *
       * @param x Vector to convert.
       * @param A Matrix into which to write conversion.
       *
       * The vector is assumed to have column-wise dense storage and must
       * have the same number of elements as the matrix.
       */
      template <class VT, class MT>
      void vectorToMatrix(const VT& x, MT& A);

      /**
       * Convert matrix to vector.
       *
       * @param A Matrix to convert.
       * @param x Vector into which to write conversion.
       *
       * The matrix is written into the vector using column-wise dense
       * storage and must have the same number of elements as the vector.
       */
      template <class MT, class VT>
      void matrixToVector(const MT& A, VT& x);

    }
  }
}

template <class T>
boost::numeric::ublas::matrix_vector_range<T> indii::ml::aux::diag(T& A) {
  /* pre-condition */
  assert (A.size1() == A.size2());

  const boost::numeric::ublas::range step(0, A.size1());

  return boost::numeric::ublas::matrix_vector_range<T>(A, step, step);
}

template <class VT, class MT>
void indii::ml::aux::vectorToMatrix(const VT& x, MT& A) {
  /* pre-condition */
  assert (x.size() == A.size1() * A.size2());
  
  const unsigned int M = A.size1(), N = A.size2();
  unsigned int col;
  for (col = 0; col < N; col++) {
    column(A,col) = subrange(x, col*M, (col+1)*M);
  }
}

template <class MT, class VT>
void indii::ml::aux::matrixToVector(const MT& A, VT& x) {
  /* pre-condition */
  assert (x.size() == A.size1() * A.size2());

  const unsigned int M = A.size1(), N = A.size2();
  unsigned int col;
  for (col = 0; col < N; col++) {
    subrange(x, col*M, (col+1)*M) = column(A,col);
  }
}

#endif

