#ifndef INDII_ML_FILTER_KALMANSMOOTHER_HPP
#define INDII_ML_FILTER_KALMANSMOOTHER_HPP

#include "TwoFilterSmoother.hpp"
#include "KalmanSmootherModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Kalman two-filter smoother.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 552 $
 * @date $Date: 2008-09-04 15:25:50 +0100 (Thu, 04 Sep 2008) $
 *
 * @param T The type of time.
 *
 * The Kalman two-filter smoother performs both a forward and
 * backwards filtering pass, fusing the estimates from the two passes
 * into the smoothed estimate. The forwards filtering pass is
 * identical to that of KalmanFilter. The backwards filtering pass
 * requires the inverse of the state transition model, and works in an
 * analogous fashion to the forwards pass, but with time reversed.
 *
 * It is suitable for models with linear transition and measurement
 * functions, approximating state and noise with
 * indii::ml::aux::GaussianPdf distributions.
 *
 * @bug Buggy, see test results. Consider RauchTungStriebelSmoother
 * instead.
 * 
 * @see indii::ml::filter for general usage guidelines.
 * @see LinearModel for more detail on linear filters.
 */
template <class T>
class KalmanSmoother : public TwoFilterSmoother<T> {
public:
  /**
   * Create new Kalman two-filter smoother.
   *
   * @param model Model to estimate.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  KalmanSmoother(KalmanSmootherModel<T>* model,
      const T tT, const indii::ml::aux::GaussianPdf& p_xT);

  /**
   * Destructor.
   */
  virtual ~KalmanSmoother();

  virtual void smooth(const T tn, const indii::ml::aux::vector& ytn,
      const indii::ml::aux::GaussianPdf& p_xtn_ytn);

  virtual indii::ml::aux::GaussianPdf backwardMeasure();

  virtual indii::ml::aux::GaussianPdf smoothedMeasure();

private:
  /**
   * Model to estimate.
   */
  KalmanSmootherModel<T>* model;

  /**
   * Size of the state space.
   */
  const unsigned int N;
};

    }
  }
}

template <class T>
indii::ml::filter::KalmanSmoother<T>::KalmanSmoother(
    KalmanSmootherModel<T>* model, const T tT,
    const indii::ml::aux::GaussianPdf& p_xT) :
    TwoFilterSmoother<T>(tT, p_xT), model(model), N(p_xT.getDimensions()) {
  //
}

template <class T>
indii::ml::filter::KalmanSmoother<T>::~KalmanSmoother() {
  //
}

template <class T>
void indii::ml::filter::KalmanSmoother<T>::smooth(const T tn,
    const indii::ml::aux::vector& ytn,
    const indii::ml::aux::GaussianPdf& p_xtn_ytn) {
  namespace aux = indii::ml::aux;

  /* pre-condition */
  assert (tn < this->tn);

  T delta = this->tn - tn;
  this->tn = tn;

  /* backward filter */
  aux::GaussianPdf p_xtn_ytnp1_b(model->p_xtnm1_ytn(this->p_xtn_ytn_b,
      delta));
  this->p_xtn_ytn_b = model->p_xtnm1_ytnm1(p_xtn_ytnp1_b, ytn, delta);

  /* fuse p_xtn_ytn and p_xtn_ytnp1_b */
  const aux::symmetric_matrix& P_xtn_ytn = p_xtn_ytn.getCovariance();
  const aux::symmetric_matrix& P_xtn_ytnp1_b = p_xtn_ytnp1_b.getCovariance();
  aux::matrix PI_xtn_ytn(N,N);
  aux::matrix PI_xtn_ytnp1_b(N,N);
  aux::matrix X(N,N);

  /* calculate inverses */
  noalias(X) = P_xtn_ytn;
  aux::inv(X, PI_xtn_ytn);

  noalias(X) = P_xtn_ytnp1_b;
  aux::inv(X, PI_xtn_ytnp1_b);

  /* calculate fused covariance */
  aux::symmetric_matrix P_xtn_ytT(N);
  aux::matrix Y(N,N);
  noalias(Y) = PI_xtn_ytn + PI_xtn_ytnp1_b;
  /**
   * @todo ^^^ Subtract sigma_t^-1, see Jordan 15.7.2 end.
   */
  aux::inv(Y, X);
  noalias(P_xtn_ytT) = X;

  /* calculate fused mean */
  const aux::vector& x_xtn_ytn = p_xtn_ytn.getExpectation();
  const aux::vector& x_xtn_ytnp1_b = p_xtn_ytnp1_b.getExpectation();
  aux::vector x_xtn_ytT(N);

  noalias(x_xtn_ytT) = prod(PI_xtn_ytn, x_xtn_ytn);
  noalias(x_xtn_ytT) += prod(PI_xtn_ytnp1_b, x_xtn_ytnp1_b);
  x_xtn_ytT = prod(P_xtn_ytT, x_xtn_ytT);

  /* update smoothed state */
  this->p_xtn_ytT.setExpectation(x_xtn_ytT);
  this->p_xtn_ytT.setCovariance(P_xtn_ytT);

  /* post-condition */
  assert (this->tn == tn);
}

template <class T>
indii::ml::aux::GaussianPdf
    indii::ml::filter::KalmanSmoother<T>::smoothedMeasure() {
  return model->p_y_x(this->p_xtn_ytT);
}

template <class T>
indii::ml::aux::GaussianPdf
    indii::ml::filter::KalmanSmoother<T>::backwardMeasure() {
  return model->p_y_x(this->p_xtn_ytn_b);
}

#endif
