#ifndef INDII_ML_FILTER_UNSCENTEDKALMANSMOOTHER_HPP
#define INDII_ML_FILTER_UNSCENTEDKALMANSMOOTHER_HPP

#include "UnscentedKalmanFilter.hpp"
#include "TwoFilterSmoother.hpp"

namespace indii {
  namespace ml {
    namespace filter {

      template<class T> class UnscentedKalmanSmootherModel;
      template<class T> class UnscentedKalmanSmootherBackwardTransitionAdaptor;

/**
 * Unscented Kalman two-filter smoother.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * The Unscented Kalman two-filter smoother performs both a forward
 * and backwards filtering pass, fusing the estimates from the two
 * passes into the smoothed estimate. The forwards filtering pass is
 * identical to that of UnscentedKalmanFilter. The backwards filtering
 * pass requires the inverse of the state transition model, and works
 * in an analogous fashion to the forwards pass, but with time
 * reversed.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T>
class UnscentedKalmanSmoother : public TwoFilterSmoother<T> {

  friend class UnscentedKalmanSmootherBackwardTransitionAdaptor<T>;

public:
  /**
   * Create new Unscented Kalman two-filter smoother.
   *
   * @param model Model to estimate.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  UnscentedKalmanSmoother(UnscentedKalmanSmootherModel<T>* model,
      const T tT,
      const indii::ml::aux::GaussianPdf& p_xT);

  /**
   * Destructor.
   */
  virtual ~UnscentedKalmanSmoother();

  virtual void smooth(const T tn, const indii::ml::aux::vector& ytn,
      const indii::ml::aux::GaussianPdf& p_xtn_ytn);

  virtual indii::ml::aux::GaussianPdf backwardMeasure();

  virtual indii::ml::aux::GaussianPdf smoothedMeasure();

  /**
   * Get the unscented transformation used for the backward transition
   * function. This allows its parameters to be adjusted.
   */
  UnscentedTransformation<T>* getBackwardTransitionTransformation();

  /**
   * Get the unscented transformation used for the measurement
   * function. This allows its parameters to be adjusted.
   */
  UnscentedTransformation<T>* getMeasurementTransformation();

protected:
  /**
   * Size of the state space.
   */
  const unsigned int N;

  /**
   * Size of the system noise space.
   */
  const unsigned int W;

  /**
   * Size of the measurement noise space.
   */
  const unsigned int V;

  /**
   * \f$p(\mathcal{X}_n\,|\,\mathbf{y}_{n:T})\f$; backward filter density.
   */
  aux::GaussianPdf p_Xtn_ytn_b;

  /**
   * Model to estimate.
   */
  UnscentedKalmanSmootherModel<T>* model;

  /**
   * Unscented transformation for backward transition function.
   */
  UnscentedTransformationModel<T>* backwardTransitionModel;

  /**
   * Unscented transformation for backward transition function.
   */
  UnscentedTransformation<T>* backwardTransitionTransform;

  /**
   * Unscented transformation for measurement function.
   */
  UnscentedTransformationModel<T>* measurementModel;

  /**
   * Unscented transformation for measurement function.
   */
  UnscentedTransformation<T>* measurementTransform;

};

    }
  }
}

#include "UnscentedKalmanSmootherBackwardTransitionAdaptor.hpp"

#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

template <class T>
indii::ml::filter::UnscentedKalmanSmoother<T>::UnscentedKalmanSmoother(
    UnscentedKalmanSmootherModel<T>* model, const T tT,
    const aux::GaussianPdf& p_xT) :
    TwoFilterSmoother<T>(tT, p_xT),
    N(model->getStateSize()),
    W(model->getSystemNoise().getDimensions()),
    V(model->getMeasurementNoise().getDimensions()),
    p_Xtn_ytn_b(N+W+V),
    model(model) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  /* pre-condition */
  assert (p_xT.getDimensions() == model->getStateSize());

  /* unscented transformations */
  backwardTransitionModel =
      new UnscentedKalmanSmootherBackwardTransitionAdaptor<T>(model);
  backwardTransitionTransform = new UnscentedTransformation<T>(
      *backwardTransitionModel);
  measurementModel = new UnscentedKalmanFilterMeasurementAdaptor<T>(model);
  measurementTransform = new UnscentedTransformation<T>(*measurementModel);

  /* ranges for subvectors and submatrices */
  ublas::range xRange(0,N);
  ublas::range wRange(N,N+W);
  ublas::range vRange(N+W,N+W+V);

  /* initialise augmented state */
  aux::vector mu(N+W+V);
  aux::symmetric_matrix sigma(N+W+V);
  aux::GaussianPdf& w = model->getSystemNoise();
  aux::GaussianPdf& v = model->getMeasurementNoise();

  noalias(project(mu, xRange)) = p_xT.getExpectation();
  noalias(project(mu, wRange)) = w.getExpectation();
  noalias(project(mu, vRange)) = v.getExpectation();

  noalias(project(sigma, xRange, xRange)) = p_xT.getCovariance();
  noalias(project(sigma, wRange, xRange)) = aux::zero_matrix(W, N);
  noalias(project(sigma, wRange, wRange)) = w.getCovariance();
  noalias(project(sigma, vRange, xRange)) = aux::zero_matrix(V, N);
  noalias(project(sigma, vRange, wRange)) = aux::zero_matrix(V, W);
  noalias(project(sigma, vRange, vRange)) = v.getCovariance();

  p_Xtn_ytn_b.setExpectation(mu);
  p_Xtn_ytn_b.setCovariance(sigma);
}

template <class T>
indii::ml::filter::UnscentedKalmanSmoother<T>::~UnscentedKalmanSmoother() {
  delete backwardTransitionTransform;
  delete backwardTransitionModel;
  delete measurementTransform;
  delete measurementModel;
}

template <class T>
void indii::ml::filter::UnscentedKalmanSmoother<T>::smooth(const T tn,
    const indii::ml::aux::vector& ytn,
    const indii::ml::aux::GaussianPdf& p_xtn_ytn) {
  namespace aux = indii::ml::aux;
  namespace lapack = boost::numeric::bindings::lapack;
  namespace ublas = boost::numeric::ublas;

  /* pre-condition */
  assert (tn < this->tn);

  T delta = this->tn - tn;
  this->tn = tn;

  /* for convenience */
  ublas::range xRange(0,N);
  ublas::range wRange(N,N+W);
  ublas::range vRange(N+W,N+W+V);

  /* unscented transformation of augmented state through backward
     transition function */
  aux::GaussianPdf p_xtn_ytnp1_b(backwardTransitionTransform->transform(
      p_Xtn_ytn_b, delta));

  /* prepare augmented state for measurement function */
  aux::vector mu(p_Xtn_ytn_b.getExpectation());
  aux::symmetric_matrix sigma(p_Xtn_ytn_b.getCovariance());

  noalias(project(mu, xRange)) = p_xtn_ytnp1_b.getExpectation();
  noalias(project(sigma, xRange, xRange)) = p_xtn_ytnp1_b.getCovariance();

  aux::GaussianPdf p_Xtn_ytnp1(mu, sigma);

  /* unscented transformation of augmented state through measurement
     function */
  aux::matrix P_xy(N+W+V,V); // cross-correlation matrix
  aux::GaussianPdf p_ytn_Xtn(
      measurementTransform->transform(p_Xtn_ytnp1, delta, &P_xy));

  /* update p_xtn_ytn_b based on measurement */
  const aux::vector &y_ytn_Xtn = p_ytn_Xtn.getExpectation();
  const aux::symmetric_matrix &P_ytn_Xtn = p_ytn_Xtn.getCovariance();
  const aux::vector &x_Xtn_ytnp1 = p_Xtn_ytnp1.getExpectation();
  const aux::symmetric_matrix &P_Xtn_ytnp1 = p_Xtn_ytnp1.getCovariance();
  aux::matrix PI_ytn_Xtn(V,V); // inverse of P_ytn_Xtn
  aux::matrix X(P_ytn_Xtn);
  aux::inv(X, PI_ytn_Xtn);
  aux::matrix K(prod(P_xy, PI_ytn_Xtn)); // Kalman gain

  /* only state variables should be updated, so zero out entries for noise
   * variables in the Kalman gain matrix */
  noalias(project(K,wRange,ublas::range(0,V))) = aux::zero_matrix(W,V);
  noalias(project(K,vRange,ublas::range(0,V))) = aux::zero_matrix(V,V);

  noalias(mu) = x_Xtn_ytnp1 + prod(K, ytn - y_ytn_Xtn);
  X.resize(V, N+W+V, false);
  noalias(X) = prod(P_ytn_Xtn, trans(K));
  noalias(sigma) = P_Xtn_ytnp1 - prod(K, X);

  p_Xtn_ytn_b.setExpectation(mu);
  p_Xtn_ytn_b.setCovariance(sigma);
  this->p_xtn_ytn_b.setExpectation(project(mu, xRange));
  this->p_xtn_ytn_b.setCovariance(project(sigma, xRange,
      xRange));

  /* fuse p_xtn_ytn and p_xtn_ytnp1_b */
  const aux::symmetric_matrix& P_xtn_ytn= p_xtn_ytn.getCovariance();
  const aux::symmetric_matrix& P_xtn_ytnp1_b = p_xtn_ytnp1_b.getCovariance();
  aux::matrix PI_xtn_ytn(N,N);
  aux::matrix PI_xtn_ytnp1_b(N,N);
  X.resize(N,N,false);

  /* calculate inverses */
  noalias(X) = P_xtn_ytn;
  aux::inv(X, PI_xtn_ytn);

  noalias(X) = P_xtn_ytnp1_b;
  aux::inv(X, PI_xtn_ytnp1_b);

  /* calculate fused covariance */
  aux::symmetric_matrix P_xtn_ytT(N);
  aux::matrix Y(N,N);
  noalias(Y) = PI_xtn_ytn + PI_xtn_ytnp1_b;

  aux::inv(Y,X);
  noalias(P_xtn_ytT) = ublas::symmetric_adaptor<aux::matrix, ublas::lower>(X);

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
    indii::ml::filter::UnscentedKalmanSmoother<T>::backwardMeasure() {
  return measurementTransform->transform(p_Xtn_ytn_b);
}

template <class T>
indii::ml::aux::GaussianPdf
    indii::ml::filter::UnscentedKalmanSmoother<T>::smoothedMeasure() {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  /* for convenience */
  ublas::range xRange(0,N);
  ublas::range wRange(N,N+W);
  ublas::range vRange(N+W,N+W+V);

  /* prepare augmented state for measurement function */
  aux::vector mu(p_Xtn_ytn_b.getExpectation());
  aux::symmetric_matrix sigma(p_Xtn_ytn_b.getCovariance());

  noalias(project(mu, xRange)) = this->p_xtn_ytT.getExpectation();
  noalias(project(sigma, xRange, xRange)) =
      this->p_xtn_ytT.getCovariance();

  aux::GaussianPdf p_Xtn_ytT(mu, sigma);

  return measurementTransform->transform(p_Xtn_ytT);
}

template <class T>
indii::ml::filter::UnscentedTransformation<T>*
    indii::ml::filter::UnscentedKalmanSmoother<T>::getBackwardTransitionTransformation() {
  return backwardTransitionTransform;
}

template <class T>
indii::ml::filter::UnscentedTransformation<T>*
    indii::ml::filter::UnscentedKalmanSmoother<T>::getMeasurementTransformation() {
  return measurementTransform;
}

#endif

