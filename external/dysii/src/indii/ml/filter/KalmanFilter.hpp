#ifndef INDII_ML_ESTIMATOR_KALMANFILTER_HPP
#define INDII_ML_ESTIMATOR_KALMANFILTER_HPP

#include "../aux/vector.hpp"
#include "../aux/GaussianPdf.hpp"

#include "Filter.hpp"
#include "KalmanFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Kalman %filter.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 * 
 * KalmanFilter is suitable for models with linear transition and
 * measurement functions, approximating state and noise with
 * indii::ml::aux::GaussianPdf distributions.
 *
 * @see indii::ml::filter for general usage guidelines.
 * @see LinearModel for more detail on linear filters.
 */
template <class T = unsigned int>
class KalmanFilter : public Filter<T> {
public:
  /**
   * Create new Kalman %filter.
   *
   * @param model Model to estimate.
   * @param p_x0 \f$p(\mathbf{x}_0)\f$; prior over the
   * initial state \f$\mathbf{x}_0\f$.
   */
  KalmanFilter(KalmanFilterModel<T>* model,
      const indii::ml::aux::GaussianPdf& p_x0);

  /**
   * Destructor.
   */
  virtual ~KalmanFilter();

  virtual void filter(const T tnp1, const indii::ml::aux::vector& ytnp1);

  virtual indii::ml::aux::GaussianPdf measure();

private:
  /**
   * Model to estimate.
   */
  KalmanFilterModel<T>* model;

};

    }
  }
}

namespace aux = indii::ml::aux;

using namespace indii::ml::filter;

template <class T>
KalmanFilter<T>::KalmanFilter(KalmanFilterModel<T>* model,
    const aux::GaussianPdf& p_x0) : Filter<T>(p_x0),
    model(model) {
  //
}

template <class T>
KalmanFilter<T>::~KalmanFilter() {
  //
}

template <class T>
void KalmanFilter<T>::filter(T tnp1, const aux::vector& ytnp1) {
  /* pre-condition */
  assert (tnp1 > Filter<T>::tn);

  /* update time */
  T delta = tnp1 - Filter<T>::tn;
  Filter<T>::tn = tnp1;

  /* system update */
  aux::GaussianPdf p_xtnp1_ytn(model->p_xtnp1_ytn(Filter<T>::p_xtn_ytn,delta));

  /* measurement update */
  Filter<T>::p_xtn_ytn = model->p_xtnp1_ytnp1(p_xtnp1_ytn, ytnp1, delta);

  /* post-condition */
  assert (Filter<T>::tn == tnp1);
}

template <class T>
aux::GaussianPdf KalmanFilter<T>::measure() {
  return model->p_y_x(Filter<T>::p_xtn_ytn);
}

#endif
