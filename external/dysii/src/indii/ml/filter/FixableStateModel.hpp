#ifndef INDII_ML_FILTER_FIXABLESTATEMODEL_HPP
#define INDII_ML_FILTER_FIXABLESTATEMODEL_HPP

#include "../aux/vector.hpp"
#include "../aux/matrix.hpp"
#include "../aux/DiracMixturePdf.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Model with fixable state variables.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Version$
 * @date $Date: 2008-08-16 16:59:47 +0100 (Sat, 16 Aug 2008) $
 *
 * Allows one or more state variables to be fixed. Fixing a variable removes
 * it from the state, reducing the state size by one. The state may then be
 * projected up into the full space of both state and fixed variables using
 * the fromState() method, and back again using the toState() method.
 *
 * The main purpose of this class is to allow the fixing of parameters, which
 * is useful for diagnostics and model testing, with minimal changes to a
 * model. For example, the transition() method of a ParticleFilterModel
 * derived class can apply fromState() to the particle @c s passed to it to
 * standardise its representation regardless of which variables are fixed,
 * transition the standardised particle, then apply toState() before
 * returning the result.
 */
class FixableStateModel {
public:
  /**
   * Default constructor for restoring from serialization.
   */
  FixableStateModel();

  /**
   * Constructor.
   * 
   * @param N Initial state size. One or more components of the state may
   * subsequently be fixed and the state size will be reduced.
   */
  FixableStateModel(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~FixableStateModel();

  /**
   * Get number of non-fixed variables.
   */
  virtual unsigned int getVariableSize() const;

  /**
   * Get number of fixed variables.
   */
  virtual unsigned int getFixedSize() const;

  /**
   * Fix the value of a variable. If the variable is already fixed, its value
   * is updated to the new value given.
   *
   * @param i The index of the variable amongst both the fixed and state
   * variables.
   * @param value Value to which to fix the variable.
   *
   * @deprecated Use fix()
   */
  void fix(const unsigned int i, const double value);

  /**
   * Is the value of a particular variable fixed?
   *
   * @param i The index of the variable amongst both the fixed and state
   * variables.
   *
   * @return True if the value @p i is fixed, false otherwise.
   */
  bool isFixed(const unsigned int i) const;
  
  /**
   * Get the value to which a particular variable is fixed.
   *
   * @param i The index of the variable amongst both the fixed and state
   * variables.
   *
   * @return The value of the variable if fixed, undefined if not fixed.
   */
  double getFixedValue(const unsigned int i) const;

  /**
   * Project vector of both fixed and state variables to state variables
   * only.
   *
   * @param x Vector of both fixed and state variables.
   *
   * @return Vector of state variables only.
   */
  indii::ml::aux::vector condense(const indii::ml::aux::vector& x) const;

  /**
   * Project vector of state variables into state and fixed variables.
   *
   * @param x Vector of state variables only.
   *
   * @return Vector of both fixed and state variables.
   */
  indii::ml::aux::vector expand(const indii::ml::aux::vector& x) const;

  /**
   * Project vector of both fixed and state variables to state variables
   * only.
   *
   * @param x Vector of both fixed and state variables.
   *
   * @return Vector of state variables only.
   */
  indii::ml::aux::DiracPdf condense(const indii::ml::aux::DiracPdf& x) const;

  /**
   * Project vector of state variables into state and fixed variables.
   *
   * @param x Vector of state variables only.
   *
   * @return Vector of both fixed and state variables.
   */
  indii::ml::aux::DiracPdf expand(const indii::ml::aux::DiracPdf& x) const;

  /**
   * Project matrix (e.g. covariance matrix) of both fixed and
   * state variables to state variables only.
   *
   * @param x Matrix of both fixed and state variables.
   *
   * @return Matrix of state variables only.
   */
  template <class M>
  M condense(const M& x) const;

  /**
   * Project matrix (e.g. covariance matrix) of state variables
   * only to both fixed and state variables.
   *
   * @param x Matrix of state variables only.
   *
   * @return Matrix of both fixed and state variables.
   */
  template <class M>
  M expand(const M& x) const;

private:
  /**
   * Size of state.
   */
  unsigned int N;
  
  /**
   * Number of fixed variables.
   */
  unsigned int F;
  
  /**
   * Fixed variable values. Zero for all others.
   */
  indii::ml::aux::sparse_vector fixed;

  /**
   * Projection of fixed and state variables to state variables only.
   */
  indii::ml::aux::projection_matrix projectCondense;

  /**
   * Serialize, or restore from serialization.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  
};

    }
  }
}

#include "boost/numeric/ublas/operation.hpp"
#include "boost/numeric/ublas/operation_sparse.hpp"

inline bool indii::ml::filter::FixableStateModel::isFixed(
    const unsigned int i) const {
  bool result = true;
  unsigned int row;
  
  for (row = 0; row < projectCondense.size1(); row++) {
    if (projectCondense(row,i) == 1) {
      result = false;
      break;
    }
  }

  return result;
}

inline double indii::ml::filter::FixableStateModel::getFixedValue(
    const unsigned int i) const {
  /* pre-condition */
  assert (i < N + F);
    
  return fixed(i);
}

inline indii::ml::aux::vector indii::ml::filter::FixableStateModel::condense(
    const aux::vector& x) const {
  /* pre-condition */
  assert (x.size() == N + F);

  namespace ublas = boost::numeric::ublas;

  if (F == 0) {
    return x;
  } else {
    aux::vector result(N);
    ublas::axpy_prod(projectCondense, x, result, true);
    return result;
  }
}

inline indii::ml::aux::vector indii::ml::filter::FixableStateModel::expand(
    const aux::vector& x) const {
  /* pre-condition */
  assert (x.size() == N);

  namespace ublas = boost::numeric::ublas;

  if (F == 0) {
    return x;
  } else {
    aux::vector result(fixed);
    ublas::axpy_prod(x, projectCondense, result, false);
    return result;
  }
}

inline indii::ml::aux::DiracPdf
    indii::ml::filter::FixableStateModel::condense(const aux::DiracPdf& x)
     const {
  /* pre-condition */
  assert (x.size() == N + F);

  namespace ublas = boost::numeric::ublas;

  if (F == 0) {
    return x;
  } else {
    aux::DiracPdf result(N);
    ublas::axpy_prod(projectCondense, x, result, true);
    return result;
  }
}

inline indii::ml::aux::DiracPdf indii::ml::filter::FixableStateModel::expand(
    const aux::DiracPdf& x) const {
  /* pre-condition */
  assert (x.size() == N);

  namespace ublas = boost::numeric::ublas;

  if (F == 0) {
    return x;
  } else {
    aux::DiracPdf result(fixed);
    ublas::axpy_prod(x, projectCondense, result, false);
    return result;
  }
}

template <class M>
inline M indii::ml::filter::FixableStateModel::condense(const M& x) const {
  /* pre-condition */
  assert (x.size1() == N + F);

  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  if (F == 0) {
    return x;
  } else {
    aux::matrix tmp(prod(projectCondense, x));
    M result(prod(tmp, trans(projectCondense)));
    return result;
  }
}

template <class M>
inline M indii::ml::filter::FixableStateModel::expand(const M& x) const {
  /* pre-condition */
  assert (x.size1() == N);

  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  if (F == 0) {
    return x;
  } else {
    aux::matrix tmp(prod(trans(projectCondense), x));
    M result(prod(tmp, projectCondense));
    return result;
  }
}

template<class Archive>
void indii::ml::filter::FixableStateModel::serialize(Archive& ar,
    const unsigned int version) {
  ar & N;
  ar & F;
  ar & fixed;
  ar & projectCondense;
}

#endif

