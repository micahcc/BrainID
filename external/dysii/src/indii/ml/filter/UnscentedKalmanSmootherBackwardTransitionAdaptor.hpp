#ifndef INDII_ML_FILTER_UNSCENTEDKALMANSMOOTHERBACKWARDTRANSITIONADAPTOR_HPP
#define INDII_ML_FILTER_UNSCENTEDKALMANSMOOTHERBACKWARDTRANSITIONADAPTOR_HPP

#include "UnscentedTransformationModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

  template <class T> class UnscentedTransformation;

/**
 * Adaptor mapping UnscentedTransformationModel interface to method calls in
 * UnscentedKalmanSmootherModel.
 * 
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * For internal use only.
 */
template <class T = unsigned int>
class UnscentedKalmanSmootherBackwardTransitionAdaptor :
    public UnscentedTransformationModel<T> {

  friend class UnscentedTransformation<T>;
  friend class UnscentedKalmanSmoother<T>;

private:
  /**
   * Constructor.
   * 
   * @param smoother UnscentedKalmanSmoother for which to map calls.
   */
  UnscentedKalmanSmootherBackwardTransitionAdaptor(
      UnscentedKalmanSmootherModel<T>* model);

  /**
   * Maps call to UnscentedKalmanSmootherModel::backwardTransition.
   */
  virtual indii::ml::aux::vector propagate(const indii::ml::aux::vector& x,
      const T delta = 0);

  /**
   * Smoother for which to act as adaptor.
   */
  UnscentedKalmanSmootherModel<T>* model;

};

    }
  }
}

#include "UnscentedKalmanSmootherModel.hpp"

template <class T>
indii::ml::filter::UnscentedKalmanSmootherBackwardTransitionAdaptor<T>::UnscentedKalmanSmootherBackwardTransitionAdaptor(
    UnscentedKalmanSmootherModel<T>* model) : model(model) {
  //
}

template <class T>
indii::ml::aux::vector
    indii::ml::filter::UnscentedKalmanSmootherBackwardTransitionAdaptor<T>::propagate(
    const indii::ml::aux::vector& X, T delta) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;
    
  const unsigned int N = model->getStateSize();
  const unsigned int W = model->getSystemNoise().getDimensions();

  aux::vector x(project(X, ublas::range(0,N)));
  aux::vector w(project(X, ublas::range(N,N+W)));

  return model->backwardTransition(x, w, delta);
}

#endif
