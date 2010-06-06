#ifndef INDII_ML_ESTIMATOR_UNSCENTEDKALMANFILTER_HPP
#define INDII_ML_ESTIMATOR_UNSCENTEDKALMANFILTER_HPP

#include "../aux/vector.hpp"
#include "../aux/matrix.hpp"
#include "../aux/GaussianPdf.hpp"

#include "Filter.hpp"
#include "UnscentedTransformation.hpp"

namespace indii {
  namespace ml {
    namespace filter {

      template <class T> class UnscentedKalmanFilterModel;
      template <class T> class UnscentedKalmanFilterTransitionAdaptor;
      template <class T> class UnscentedKalmanFilterMeasurementAdaptor;

/**
 * Unscented Kalman %filter.
 * 
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * UnscentedKalmanFilter is suitable for systems with nonlinear
 * transition and measurement functions, approximating state and noise
 * with indii::ml::aux::GaussianPdf distributions.
 *
 * @see UnscentedTransformation
 * @see indii::ml::filter for general usage guidelines.
 */
template<class T = unsigned int>
class UnscentedKalmanFilter : public Filter<T> {

  friend class UnscentedKalmanFilterTransitionAdaptor<T>;
  friend class UnscentedKalmanFilterMeasurementAdaptor<T>;

public:
  /**
   * Create new unscented Kalman %filter.
   *
   * @param model Model to estimate.
   * @param p_x0 \f$p(\mathbf{x}_0)\f$; prior over the
   * initial state \f$\mathbf{x}(0)\f$.
   */
  UnscentedKalmanFilter(UnscentedKalmanFilterModel<T>* model,
      const indii::ml::aux::GaussianPdf& p_x0);

  /**
   * Destructor.
   */
  virtual ~UnscentedKalmanFilter();

  /**
   * Get the unscented transformation used for the transition
   * function. This allows its parameters to be adjusted.
   */
  UnscentedTransformation<T>* getTransitionTransformation();

  /**
   * Get the unscented transformation used for the measurement
   * function. This allows its parameters to be adjusted.
   */
  UnscentedTransformation<T>* getMeasurementTransformation();

  virtual void filter(const T tnp1, const indii::ml::aux::vector& ytnp1);

  virtual indii::ml::aux::GaussianPdf measure();

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
   * \f$p(\mathcal{X}_n\,|\,\mathbf{y}_{1:n})\f$; filter density over
   * augmented state.
   */
  indii::ml::aux::GaussianPdf p_Xtn_ytn;

  /**
   * Model.
   */
  UnscentedKalmanFilterModel<T>* model;

  /**
   * Unscented transformation for transition function.
   */
  UnscentedTransformationModel<T>* transitionModel;

  /**
   * Unscented transformation for measurement function.
   */
  UnscentedTransformationModel<T>* measurementModel;

  /**
   * Unscented transformation for transition function.
   */
  UnscentedTransformation<T>* transitionTransform;

  /**
   * Unscented transformation for measurement function.
   */
  UnscentedTransformation<T>* measurementTransform;

};

    }
  }
}

#include "UnscentedKalmanFilterTransitionAdaptor.hpp"
#include "UnscentedKalmanFilterMeasurementAdaptor.hpp"
#include "UnscentedKalmanFilterModel.hpp"

#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_symmetric.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

template <class T>
indii::ml::filter::UnscentedKalmanFilter<T>::UnscentedKalmanFilter(
    indii::ml::filter::UnscentedKalmanFilterModel<T>* model,
    const indii::ml::aux::GaussianPdf& p_x0) :
    Filter<T>(p_x0),
    N(model->getStateSize()),
    W(model->getSystemNoise().getDimensions()),
    V(model->getMeasurementNoise().getDimensions()),
    p_Xtn_ytn(N+W+V),
    model(model) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  /* pre-condition */
  assert (p_x0.getDimensions() == model->getStateSize());
    
  /* unscented transformations */
  transitionModel = new UnscentedKalmanFilterTransitionAdaptor<T>(model);
  transitionTransform = new UnscentedTransformation<T>(*transitionModel);
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

  noalias(project(mu, xRange)) = p_x0.getExpectation();
  noalias(project(mu, wRange)) = w.getExpectation();
  noalias(project(mu, vRange)) = v.getExpectation();

  noalias(project(sigma, xRange, xRange)) = p_x0.getCovariance();
  noalias(project(sigma, wRange, xRange)) = aux::zero_matrix(W, N);
  noalias(project(sigma, wRange, wRange)) = w.getCovariance();
  noalias(project(sigma, vRange, xRange)) = aux::zero_matrix(V, N);
  noalias(project(sigma, vRange, wRange)) = aux::zero_matrix(V, W);
  noalias(project(sigma, vRange, vRange)) = v.getCovariance();

  p_Xtn_ytn.setExpectation(mu);
  p_Xtn_ytn.setCovariance(sigma);
}

template <class T>
indii::ml::filter::UnscentedKalmanFilter<T>::~UnscentedKalmanFilter() {
  delete transitionTransform;
  delete measurementTransform;
  delete transitionModel;
  delete measurementModel;
}

template <class T>
void indii::ml::filter::UnscentedKalmanFilter<T>::filter(const T tnp1,
    const indii::ml::aux::vector& ytnp1) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;
  namespace lapack = boost::numeric::bindings::lapack;

  /* pre-condition */
  assert (tnp1 > this->tn);
  assert (ytnp1.size() % V == 0);

  const T delta = tnp1 - this->tn;
  this->tn = tnp1;

  /* ranges for subvectors and submatrices */
  ublas::range xRange(0,N);
  ublas::range wRange(N,N+W);
  ublas::range vRange(N+W,N+W+V);

  /* unscented transformation of augmented state through transition
     function */
  aux::GaussianPdf p_xtnp1_ytn(transitionTransform->transform(p_Xtn_ytn,
      delta));

  /* prepare augmented state for measurement function */
  aux::vector mu(p_Xtn_ytn.getExpectation());
  aux::symmetric_matrix sigma(p_Xtn_ytn.getCovariance());

  noalias(project(mu, xRange)) = p_xtnp1_ytn.getExpectation();
  noalias(project(sigma, xRange, xRange)) = p_xtnp1_ytn.getCovariance();

  aux::GaussianPdf p_Xtnp1_ytn(mu, sigma);

  /* unscented transformation of augmented state through measurement
     function */
  aux::matrix P_xy(N+W+V,V); // cross-correlation matrix
  aux::GaussianPdf p_ytnp1_Xtnp1(measurementTransform->transform(p_Xtnp1_ytn,
      delta, &P_xy));

  /* update p_xtn_ytn based on measurement */
  const aux::vector &y_ytnp1_Xtnp1 = p_ytnp1_Xtnp1.getExpectation();
  const aux::symmetric_matrix &P_ytnp1_Xtnp1 = p_ytnp1_Xtnp1.getCovariance();
  const aux::vector &x_Xtnp1_ytn = p_Xtnp1_ytn.getExpectation();
  const aux::symmetric_matrix &P_Xtnp1_ytn = p_Xtnp1_ytn.getCovariance();

  /* 
   * Calculate Kalman gain directly without intermediate calculation
   * of inverse, improves performance marginally.
   *
   * Calculate Kalman gain. <tt>K = P_xy *
   * inv(P_ytnp1_Xtnp1)</tt>. Calculate as <tt>trans(K) =
   * inv(P_ytnp1_Xtnp1) * trans(P_xy), so that LAPACK gesv function
   * can be used, noting that <tt>trans(P_ytnp1_Xtnp1) ==
   * P_ytnp1_Xtnp1</tt> as <tt>P_ytnp1_Xtnp1</tt> is symmetric.
   *
   * Note that Kalman gain applies only to state variables, not noise
   * variables, so means and covariances can be projected down to only
   * state variables for the update.
   */
  aux::matrix X(P_ytnp1_Xtnp1);
  aux::matrix KT(trans(project(P_xy,xRange,ublas::range(0,V))));
  int err;
  err = lapack::gesv(X,KT);
  assert (err == 0);
  aux::matrix K(trans(KT));

  noalias(project(mu,xRange)) = project(x_Xtnp1_ytn,xRange) +
      prod(K, ytnp1 - y_ytnp1_Xtnp1);

  X.resize(P_ytnp1_Xtnp1.size1(), KT.size2(), false);
  noalias(X) = prod(P_ytnp1_Xtnp1, KT);
  noalias(project(sigma,xRange,xRange)) = project(P_Xtnp1_ytn,xRange,xRange)
      - prod(K, X);

  p_Xtn_ytn.setExpectation(mu);
  p_Xtn_ytn.setCovariance(sigma);
  this->p_xtn_ytn.setExpectation(project(mu, xRange));
  this->p_xtn_ytn.setCovariance(project(sigma, xRange, xRange));

  /* post-condition */
  assert (this->tn == tnp1);
}

template <class T>
indii::ml::aux::GaussianPdf
    indii::ml::filter::UnscentedKalmanFilter<T>::measure() {
  return measurementTransform->transform(p_Xtn_ytn);
}

template <class T>
indii::ml::filter::UnscentedTransformation<T>*
    indii::ml::filter::UnscentedKalmanFilter<T>::getTransitionTransformation() {
  return transitionTransform;
}

template <class T>
indii::ml::filter::UnscentedTransformation<T>*
    indii::ml::filter::UnscentedKalmanFilter<T>::getMeasurementTransformation() {
  return measurementTransform;
}

#endif
