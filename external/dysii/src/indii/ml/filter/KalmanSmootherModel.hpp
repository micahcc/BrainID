#ifndef INDII_ML_FILTER_KALMANSMOOTHERMODEL_HPP
#define INDII_ML_FILTER_KALMANSMOOTHERMODEL_HPP

#include "KalmanFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * KalmanSmoother compatible model.
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
class KalmanSmootherModel : virtual public KalmanFilterModel<T> {
public:
  /**
   * Destructor.
   */
  virtual ~KalmanSmootherModel() = 0;

  /**
   * Predict previous system state.
   *
   * @param p_xtn_ytn \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_n),\ldots,\mathbf{y}(t_T)\big)\f$; distribution
   * over states at the current time given present and future
   * measurements.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$P\big(\mathbf{x}(t_n - \Delta t)\, |
   * \,\mathbf{y}(t_n),\ldots,\mathbf{y}(t_T)\big)\f$; predicted
   * distribution over states at time \f$t_n - \Delta t\f$ given
   * future measurements.
   */
  virtual indii::ml::aux::GaussianPdf p_xtnm1_ytn(
      const indii::ml::aux::GaussianPdf& p_xtn_ytn,
      const T delta) = 0;

  /**
   * Refine prediction of previous system state using previous
   * measurement.
   *
   * @param p_xtnm1_ytn \f$P\big(\mathbf{x}(t_n - \Delta
   * t)\,|\,\mathbf{y}(t_n),\ldots,\mathbf{y}(t_T)\big)\f$; predicted
   * distribution over states at time \f$t_n - \Delta t\f$ given the
   * history of measurements. Typically obtained from prior call to
   * #p_xtnm1_ytn.
   * @param ytnm1 \f$\mathbf{y}(t_n - \Delta t)\f$; the measurement at
   * time \f$t_n - \Delta t\f$.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$P\big(\mathbf{x}(t_n - \Delta t)\, | \,\mathbf{y}(t_n
   * - \Delta),\ldots,\mathbf{y}(t_T)\big)\f$; distribution over
   * states at time \f$t_n - \Delta t\f$ given the present and future
   * measurements.
   */
  virtual indii::ml::aux::GaussianPdf p_xtnm1_ytnm1(
      const indii::ml::aux::GaussianPdf& p_xtnm1_ytn,
      const indii::ml::aux::vector& ytnm1, const T delta) = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::KalmanSmootherModel<T>::~KalmanSmootherModel() {
  //
}

#endif
