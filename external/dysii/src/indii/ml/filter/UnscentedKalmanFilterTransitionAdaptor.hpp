#ifndef INDII_ML_FILTER_UNSCENTEDKALMANFILTERTRANSITIONADAPTOR_HPP
#define INDII_ML_FILTER_UNSCENTEDKALMANFILTERTRANSITIONADAPTOR_HPP

#include "UnscentedTransformationModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

  template <class T> class UnscentedTransformation;
  template <class T> class UnscentedKalmanFilter;
  template <class T> class UnscentedKalmanSmoother;
  template <class T> class UnscentedKalmanFilterModel;

/**
 * Adaptor mapping UnscentedTransformationModel interface to method calls in
 * UnscentedKalmanFilterModel.
 * 
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * For internal use only.
 */
template <class T>
class UnscentedKalmanFilterTransitionAdaptor :
    public UnscentedTransformationModel<T> {

  friend class UnscentedTransformation<T>;
  friend class UnscentedKalmanFilter<T>;
  friend class UnscentedKalmanSmoother<T>;

private:
  /**
   * Constructor.
   * 
   * @param filter UnscentedKalmanFilter for which to map calls.
   */
  UnscentedKalmanFilterTransitionAdaptor(
      UnscentedKalmanFilterModel<T>* model);

  /**
   * Maps call to UnscentedKalmanFilterModel::transition.
   */
  virtual indii::ml::aux::vector propagate(const indii::ml::aux::vector& x,
      const T delta = 0);

  /**
   * Filter for which to act as adaptor.
   */
  UnscentedKalmanFilterModel<T>* model;

};

    }
  }
}

#include "UnscentedKalmanFilterModel.hpp"

template <class T>
indii::ml::filter::UnscentedKalmanFilterTransitionAdaptor<T>::UnscentedKalmanFilterTransitionAdaptor(
    UnscentedKalmanFilterModel<T>* model) : model(model) {
  //
}

template <class T>
indii::ml::aux::vector
    indii::ml::filter::UnscentedKalmanFilterTransitionAdaptor<T>::propagate(
    const indii::ml::aux::vector& X, const T delta) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;
    
  const unsigned int N = model->getStateSize();
  const unsigned int W = model->getSystemNoise().getDimensions();

  aux::vector x(project(X, ublas::range(0,N)));
  aux::vector w(project(X, ublas::range(N,N+W)));

  return model->transition(x, w, delta);
}

#endif

