#ifndef INDII_ML_FILTER_KALMANFILTERMODEL_HPP
#define INDII_ML_FILTER_KALMANFILTERMODEL_HPP

#include "../aux/GaussianPdf.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * KalmanFilter compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 301 $
 * @date $Date: 2007-09-10 23:56:50 +0100 (Mon, 10 Sep 2007) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class KalmanFilterModel {
public:
  /**
   * Destructor.
   */
  virtual ~KalmanFilterModel() = 0;

  /**
   * Predict next system state.
   *
   * @param p_xtn_ytn
   * \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n)\big)\f$; distribution
   * over states at the current time given the history of
   * measurements.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$P\big(\mathbf{x}(t_n + \Delta t)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n)\big)\f$; predicted
   * distribution over states at time \f$t_n + \Delta t\f$ given the
   * history of measurements.
   */
  virtual indii::ml::aux::GaussianPdf p_xtnp1_ytn(
      const indii::ml::aux::GaussianPdf& p_xtn_ytn, const T delta) = 0;

  /**
   * Refine prediction of next system state using next measurement.
   *
   * @param p_xtnp1_ytn \f$P\big(\mathbf{x}(t_n + \Delta t)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n)\big)\f$; predicted
   * distribution over states at time \f$t_n + \Delta t\f$ given the
   * history of measurements. Typically obtained from prior call to
   * #p_xtnp1_ytn.
   * @param ytnp1 \f$\mathbf{y}(t_n + \Delta t)\f$; the measurement at
   * time \f$t_n + \Delta t\f$.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$P\big(\mathbf{x}(t_n + \Delta
   * t)\,|\,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n + \Delta t)\big)\f$;
   * distribution over states at time \f$t_n + \Delta t\f$ given the
   * history of measurements and new measurement.
   */
  virtual indii::ml::aux::GaussianPdf p_xtnp1_ytnp1(
      const indii::ml::aux::GaussianPdf& p_xtnp1_ytn,
      const indii::ml::aux::vector& ytnp1, const T delta) = 0;

  /**
   * Predict measurement from system state.
   *
   * @param p_x Arbitrary distribution over system state.
   *
   * @return Distribution over measurements given the system state.
   */
  virtual indii::ml::aux::GaussianPdf p_y_x(
      const indii::ml::aux::GaussianPdf& p_x) = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::KalmanFilterModel<T>::~KalmanFilterModel() {
  //
}

#endif
