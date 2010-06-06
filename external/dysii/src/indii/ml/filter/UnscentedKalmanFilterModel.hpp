#ifndef INDII_ML_FILTER_UNSCENTEDKALMANFILTERMODEL_HPP
#define INDII_ML_FILTER_UNSCENTEDKALMANFILTERMODEL_HPP

#include "../aux/GaussianPdf.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * UnscentedKalmanFilter compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class UnscentedKalmanFilterModel {
public:
  /**
   * Destructor.
   */
  virtual ~UnscentedKalmanFilterModel() = 0;

  /**
   * Get number of dimensions in state.
   *
   * @return Number of dimensions in state.
   */
  virtual unsigned int getStateSize() = 0;

  /**
   * Get system noise.
   *
   * @return \f$\mathbf{w}\f$; system noise.
   */
  virtual indii::ml::aux::GaussianPdf& getSystemNoise() = 0;
   
  /**
   * Get measurement noise.
   *
   * @return \f$\mathbf{v}\f$; measurement noise.
   */
  virtual indii::ml::aux::GaussianPdf& getMeasurementNoise() = 0;

  /**
   * Propagate sample through the state transition function.
   *
   * @param x \f$\mathbf{x}^*\f$; state sample.
   * @param w \f$\mathbf{w}^*\f$; noise sample.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$f(\mathbf{x}^*,\mathbf{w}^*,\Delta t)\f$; propagation
   * of \f$\mathbf{x}^*\f$ through the transition function, given
   * noise of \f$\mathbf{w}^*\f$.
   */
  virtual indii::ml::aux::vector transition(const indii::ml::aux::vector& x,
      const indii::ml::aux::vector& w, T delta) = 0;

  /**
   * Propagate sample through the measurement function.
   *
   * @param x \f$\mathbf{x}^*\f$; state sample.
   * @param v \f$\mathbf{v}^*\f$; noise sample.
   *
   * @return \f$g(\mathbf{x}^*,\mathbf{v}^*)\f$; propagation of
   * \f$\mathbf{x}^*\f$ through the measurement function, given noise
   * of \f$\mathbf{v}^*\f$.
   */
  virtual indii::ml::aux::vector measure(const indii::ml::aux::vector& x,
      const indii::ml::aux::vector& v) = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::UnscentedKalmanFilterModel<T>::~UnscentedKalmanFilterModel() {
  //
}

#endif
