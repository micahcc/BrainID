#ifndef INDII_ML_FILTER_UNSCENTEDKALMANSMOOTHERMODEL_HPP
#define INDII_ML_FILTER_UNSCENTEDKALMANSMOOTHERMODEL_HPP

#include "UnscentedKalmanFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * UnscentedKalmanSmoother compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 329 $
 * @date $Date: 2007-10-16 17:10:39 +0100 (Tue, 16 Oct 2007) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class UnscentedKalmanSmootherModel : public UnscentedKalmanFilterModel<T> {
public:
  /**
   * Destructor.
   */
  virtual ~UnscentedKalmanSmootherModel() = 0;

  /**
   * Propagate sample through the backward state transition function.
   *
   * @param x \f$\mathbf{x}^*\f$; state sample.
   * @param w \f$\mathbf{w}^*\f$; noise sample.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$f(\mathbf{x}^*,\mathbf{w}^*,-\Delta t)\f$; propagation
   * of \f$\mathbf{x}^*\f$ through the backward transition function,
   * given noise of \f$\mathbf{w}^*\f$.
   */
  virtual aux::vector backwardTransition(const indii::ml::aux::vector& x,
      const indii::ml::aux::vector& w, T delta) = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::UnscentedKalmanSmootherModel<T>::~UnscentedKalmanSmootherModel() {
  //
}

#endif
