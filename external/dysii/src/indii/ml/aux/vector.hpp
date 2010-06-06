#ifndef INDII_ML_AUX_VECTOR_HPP
#define INDII_ML_AUX_VECTOR_HPP

#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector_sparse.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/storage.hpp"

namespace indii {
  namespace ml {
    namespace aux {

      /**
       * General vector.
       */
      typedef boost::numeric::ublas::vector<double,
          boost::numeric::ublas::unbounded_array<double> > vector;

      /**
       * Zero vector.
       */
      typedef boost::numeric::ublas::zero_vector<double> zero_vector;

      /**
       * Scalar vector.
       */
      typedef boost::numeric::ublas::scalar_vector<double> scalar_vector;

      /**
       * Sparse vector.
       */
      typedef boost::numeric::ublas::mapped_vector<double> sparse_vector;

      /**
       * Element-wise vector exponential.
       *
       * @param x \f$\mathbf{x}\f$; a vector.
       *
       * @return Vector \f$\mathbf{y}\f$ with each \f$y_i = \exp(x_i)\f$.
       */
      template <class T>
      vector element_exp(const T& x);
      
      /**
       * Element-wise vector square root.
       *
       * @param x \f$\mathbf{x}\f$; a vector.
       *
       * @return Vector \f$\mathbf{y}\f$ with each \f$y_i = \sqrt{x_i}\f$.
       */
      template <class T>
      vector element_sqrt(const T& x);
      
      /**
       * Element-wise vector power.
       *
       * @param a \f$\mathbf{a}\f$; a vector.
       * @param b \f$\mathbf{b}\f$; a vector.
       *
       * @return Vector \f$\mathbf{y}\f$ with each \f$y_i = a_i^{b_i}\f$.
       */
      template <class A, class B>
      vector element_pow(const A& a, const B& b);

      /**
       * Element-wise vector power.
       *
       * @param a \f$\mathbf{a}\f$; a vector.
       * @param b \f$b\f$; a scalar.
       *
       * @return Vector \f$\mathbf{y}\f$ with each \f$y_i = a_i^b\f$.
       */
      template <class A, class B>
      vector scalar_pow(const A& a, const B b);

      /**
       * Vector p-norm.
       *
       * @param P \f$p\f$; degree of the norm.
       *
       * @param x \f$\mathbf{x}\f$; a vector.
       *
       * @return \f$\|\mathbf{x}\|_p\f$; p-norm of the given vector.
       *
       * Implementation uses template metaprogramming to utilise uBLAS
       * norm_1 and norm_2 functions where possible.
       */
      template <unsigned int P, class T>
      double norm(const T& x);

      /* Omit these from documentation */
      /// @cond NORMIMPL
      template <unsigned int P, class T>
      struct normImpl {
        static double evaluate(const T& t);
      };

      template <class T>
      struct normImpl<1,T> {
        static double evaluate(const T& t);      
      };
      
      template <class T>
      struct normImpl<2,T> {
        static double evaluate(const T& t);      
      };
      /// @endcond

      /**
       * Convert vector to double[].
       *
       * @param x Vector to convert.
       * @param a Array into which to write conversion.
       *
       * The array is assumed to have a length greater than or equal
       * to the vector.
       */
      template <class T>
      void vectorToArray(const T& x, double a[]);

      /**
       * Convert double[] to vector.
       *
       * @param a Array to convert.
       * @param x Vector into which to write conversion.
       *
       * The array is assumed to have a length greater than or equal to the
       * vector.
       */
      template <class T>
      void arrayToVector(const double a[], T& x);

    }
  }
}

template <class T>
inline indii::ml::aux::vector indii::ml::aux::element_exp(const T& x) {
  unsigned int i;
  const unsigned int N = x.size();
  vector y(N);
  
  for (i = 0; i < N; i++) {
    y(i) = exp(x(i));
  }
  
  return y;
}
      
template <class T>
inline indii::ml::aux::vector indii::ml::aux::element_sqrt(const T& x) {
  unsigned int i;
  const unsigned int N = x.size();
  vector y(N);
  
  for (i = 0; i < N; i++) {
    y(i) = sqrt(x(i));
  }
  
  return y;
}
      
template <class A, class B>
inline indii::ml::aux::vector indii::ml::aux::element_pow(const A& a,
    const B& b) {
  /* pre-condition */
  assert (a.size() == b.size());

  unsigned int i;
  const unsigned int N = a.size();
  vector y(N);

  for (i = 0; i < N; i++) {
    y(i) = pow(a(i), b(i));
  }
  
  return y;
}

template <class A, class B>
inline indii::ml::aux::vector indii::ml::aux::scalar_pow(const A& a,
    const B b) {
  unsigned int i;
  const unsigned int N = a.size();
  vector y(N);
  
  for (i = 0; i < N; i++) {
    y(i) = pow(a(i), b);
  }    

  return y;
}

/// @cond NORMIMPL
template <class T>
inline double indii::ml::aux::normImpl<1,T>::evaluate(const T& x) {
  return norm_1(x);
}

template <class T>
inline double indii::ml::aux::normImpl<2,T>::evaluate(const T& x) {
  return norm_2(x);
}

template <unsigned int P, class T>
double indii::ml::aux::normImpl<P,T>::evaluate(const T& x) {
  /* pre-condition */
  assert (P > 0);

  unsigned int i;
  double norm = 0.0;
  typename T::const_iterator iter, end;
  iter = x.begin();
  end = x.end();
  
  if (P % 2 == 0) {
    /* even number, needn't worry about abs() */
    while (iter != end) {
      norm += pow(*iter, P);
      iter++;
    }
  } else {
    /* odd number, abs() all values */
    while (iter != end) {
      norm += pow(fabs(*iter), P);
      iter++;
    }
  }
  
  return pow(norm, 1.0 / P);
}

/// @endcond

template <unsigned int P, class T>
inline double indii::ml::aux::norm(const T& x) {
  return normImpl<P,T>::evaluate(x);
}

template <class T>
void indii::ml::aux::vectorToArray(const T& x, double a[]) {
  unsigned int i, len = x.size();
  for (i = 0; i < len; i++) {
    a[i] = x(i);
  }
}

template <class T>
void indii::ml::aux::arrayToVector(const double a[], T& x) {
  unsigned int i, len = x.size();
  for (i = 0; i < len; i++) {
    x(i) = a[i];
  }
}

#endif

