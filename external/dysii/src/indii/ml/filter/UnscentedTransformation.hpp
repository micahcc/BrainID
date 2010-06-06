#ifndef INDII_ML_FILTER_UNSCENTEDTRANSFORMATION
#define INDII_ML_FILTER_UNSCENTEDTRANSFORMATION

#include "../aux/vector.hpp"
#include "../aux/GaussianPdf.hpp"

#include "UnscentedTransformationModel.hpp"
#include "../ode/ParameterCollection.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Unscented transformation.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 436 $
 * @date $Date: 2008-04-28 00:23:37 +0100 (Mon, 28 Apr 2008) $
 *
 * The unscented transformation propagates a Gaussian distributed random
 * variable through a nonlinear function, estimating its new mean and
 * covariance after application of the function. This is achieved by
 * deterministic sampling of a set of <i>sigma points</i> from the
 * distribution, propagation of these points through the function and
 * recalculation of the mean and covariance.
 *
 * @section UnscentedTransformation_details Details
 *
 * As described in @ref Julier1997 "Julier & Uhlmann (1997)" and @ref Wan2000
 * "Wan & van der Merwe (2000)", let \f$L\f$ be the number of dimensions of
 * the input random variable \f$\mathbf{x}\f$, with mean
 * \f$\mathbf{\bar{x}}\f$ and covariance matrix \f$P_x\f$. Let:
 *
 * \f[\lambda = \alpha^2 (L + \kappa) - L\f]
 *
 * \f$2L + 1\f$ sigma points \f$\mathcal{X}_i\f$ are defined as:
 *
 * \f[
 * \begin{array}{lcll}
 *   \mathcal{X}_0 & = & \mathbf{\bar{x}} \\
 *   \mathcal{X}_i & = & \mathbf{\bar{x}} + \left(\sqrt{(L+\lambda)P_x}
 *     \right)_i & i = 1,\ldots,L \\
 *   \mathcal{X}_i & = & \mathbf{\bar{x}} - \left(\sqrt{(L+\lambda)P_x}
 *     \right)_{i-L} & i = L+1,\ldots,2L \\
 * \end{array}
 * \f]
 *
 * where \f$\left(\sqrt{(L+\lambda)P_x}\right)_i\f$ is the \f$i\f$th column of
 * the matrix square root (Cholesky decomposition).
 *
 * Mean weights \f$W^{(m)}_i\f$ and covariance weights \f$W^{(c)}_i\f$ are
 * defined as:
 *
 * \f[
 * \begin{array}{lclcll}
 *   W^{(m)}_0 & = & & & \lambda/(L+\lambda) \\
 *   W^{(c)}_0 & = & & & \lambda/(L+\lambda)+(1-\alpha^2+\beta) \\
 *   W^{(m)}_i & = & W^{(c)}_i & = & 1/\left(2\left(L+\lambda\right)\right) &
 *     i = 1,\ldots,2L \\
 * \end{array}
 * \f]
 *
 * Each sigma point is then propagated through the given function \f$f\f$:
 *
 * \f[
 * \begin{array}{lcll}
 *   \mathcal{Y}_i & = & f(\mathcal{X}_i) & i = 0,\ldots,2L \\
 * \end{array}
 * \f]
 *
 * Finally, the mean and covariance of \f$f(\mathbf{x})\f$ are estimated as:
 *
 * \f{eqnarray*}
 *   \mathbf{\bar{y}} & \approx & \sum_{i=0}^{2L}W^{(m)}_i\mathcal{Y}_i \\
 *   P_y & \approx & \sum_{i=0}^{2L}W^{(c)}_i
 *     (\mathcal{Y}_i-\mathbf{\bar{y}})(\mathcal{Y}_i-\mathbf{\bar{y}})^T \\
 * \f}
 *
 * and the cross-correlation matrix \f$P_{xy}\f$ as:
 *
 * \f[P_{xy} \approx \sum_{i=0}^{2L}W^{(c)}_i
 * (\mathcal{X}_i-\mathbf{\bar{x}})(\mathcal{Y}_i-\mathbf{\bar{y}})^T\f]
 *
 * @section implementation Implementation
 *
 * This particular implementation uses a modified form of the above
 * formulas so that it need not hold the complete set of sigma points
 * in memory, as required in the calculation of \f$P_y\f$ above.
 *
 * First, we rename \f$\mathbf{\bar{y}}\f$ to \f$\mathbf{\bar{y}}_m\f$;
 * the mean calculated using mean weights
 * \f$W^{(m)}_0,\ldots,W^{(m)}_{2L}\f$. We then define a second mean
 * \f$\mathbf{\bar{y}}_c\f$ calculated using covariance weights
 * \f$W^{(c)}_0,\ldots,W^{(c)}_{2L}\f$:
 *
 * \f{eqnarray*}
 *   \mathbf{\bar{y}}_c & \approx & \sum_{i=0}^{2L}W^{(c)}_i\mathcal{Y}_i
 * \f}
 *
 * Noting that \f$W^{(c)}_i = W^{(m)}_i\f$ for \f$i = 1,\ldots,2L\f$, the two
 * means need not be calculated independently.
 *
 * The calculation of the covariance \f$P_y\f$ can then be modified to:
 *
 * \f{eqnarray*}
 *   P_y & \approx & \sum_{i=0}^{2L}W^{(c)}_i
 *     (\mathcal{Y}_i-\mathbf{\bar{y}}_m)
 *     (\mathcal{Y}_i-\mathbf{\bar{y}}_m)^T \\
 *   & = & \sum_{i=0}^{2L}W^{(c)}_i
 *     (\mathcal{Y}_i\mathcal{Y}_i^T - \mathcal{Y}_i\mathbf{\bar{y}}_m^T -
 *     \mathbf{\bar{y}}_m\mathcal{Y}_i^T +
 *     \mathbf{\bar{y}}_m\mathbf{\bar{y}}_m^T) \\
 *   & = & \sum_{i=0}^{2L}W^{(c)}_i\mathcal{Y}_i\mathcal{Y}_i^T -
 *     \mathbf{\bar{y}}_c\mathbf{\bar{y}}_m^T -
 *     \mathbf{\bar{y}}_m\mathbf{\bar{y}}_c^T +
 *     \sum_{i=0}^{2L}W^{(c)}_i\mathbf{\bar{y}}_m\mathbf{\bar{y}}_m^T
 * \f}
 *
 * @section UnscentedTransformation_references References
 *
 * @anchor Julier1997
 * Julier, S.J. & Uhlmann, J.K. A New Extension of the Kalman %Filter
 * to nonlinear Systems <i>The Proceedings of AeroSense: The 11th
 * International Symposium on Aerospace/Defense Sensing, Simulation
 * and Controls, Multi Sensor Fusion, Tracking and Resource
 * Management</i>, <b>1997</b>.
 *
 * @anchor Wan2000
 * Wan, E.A. & van der Merwe, R. The Unscented Kalman %Filter for
 * Nonlinear Estimation. <i>Proceedings of IEEE Symposium on Adaptive
 * Systems for Signal Processing Communications and Control
 * (AS-SPCC)</i>, <b>2000</b>.
 */
template <class T = unsigned int>
class UnscentedTransformation : public indii::ml::ode::ParameterCollection {
public:
  /**
   * Parameter indices.
   */
  enum Parameter {
    /**
     * \f$\alpha\f$; spread of the sigma points about
     * \f$\mathbf{\bar{x}}\f$. Usually set to a small positive value.
     */
    ALPHA,

    /**
     * \f$\beta\f$; incorporates prior knowledge of the distribution of
     * \f$\mathbf{x}\f$. Value of 2 is optimal for Gaussian distributions.
     */
    BETA,

    /**
     * \f$\kappa\f$; secondary scaling parameter. Usually set to zero.
     */
    KAPPA
  };

  /**
   * Create new transformation with default parameter values. Default values
   * are obtained from UnscentedTransformationDefaults.
   *
   * @param model Model representing the function \f$f\f$. Callee claims
   * ownership.
   */
  UnscentedTransformation(UnscentedTransformationModel<T>& model);

  /**
   * Destructor.
   */
  ~UnscentedTransformation();

  /**
   * Apply the unscented transformation.
   *
   * @param x \f$P(\mathbf{x})\f$; distribution over the random variable to
   * propagate through the function.
   * @param delta \f$\Delta t\f$; length of time through which to propagate
   * the distribution, if relevant.
   * @param P If specified, the cross correlation matrix between the input and
   * output of the function is estimated using sigma points and written into
   * this matrix. The matrix should be of size \f$n \times m\f$, where \f$n\f$
   * is the dimensionality of the input space and \f$m\f$ the dimensionality
   * of the output space.
   *
   * @return \f$P(f(\mathbf{x}))\f$; distribution over the function output.
   */
  indii::ml::aux::GaussianPdf transform(const indii::ml::aux::GaussianPdf& x,
      T delta = 0, indii::ml::aux::matrix* P = NULL);

private:
  /**
   * Number of parameters
   */
  static const unsigned int NUM_PARAMETERS = 3;

  /**
   * Model defining function through which to propagate points.
   */
  UnscentedTransformationModel<T>& model;

};

    }
  }
}

#include "UnscentedTransformationDefaults.hpp"

#include <math.h>

#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_symmetric.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

using namespace indii::ml::filter;

namespace aux = indii::ml::aux;
namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

template <class T>
UnscentedTransformation<T>::UnscentedTransformation(
    UnscentedTransformationModel<T>& model) :
    ParameterCollection(NUM_PARAMETERS), model(model) {
  setParameter(ALPHA, UnscentedTransformationDefaults::ALPHA);
  setParameter(BETA, UnscentedTransformationDefaults::BETA);
  setParameter(KAPPA, UnscentedTransformationDefaults::KAPPA);
}

template <class T>
UnscentedTransformation<T>::~UnscentedTransformation() {
  //
}

template <class T>
aux::GaussianPdf UnscentedTransformation<T>::transform(
    const aux::GaussianPdf& x, T delta, aux::matrix* P) {
  /* pre-condition */
  assert (x.getDimensions() > 0);

  const unsigned int L = x.getDimensions();
  const double LAMBDA = pow(getParameter(ALPHA), 2.0) *
      (L + getParameter(KAPPA)) - L;

  /* calculate weights */
  const double W_m_0 = LAMBDA / ((double)L + LAMBDA);
  const double W_c_0 = W_m_0 + (1.0 - pow(getParameter(ALPHA), 2.0) +
      getParameter(BETA));
  const double W_m_i = 1.0 / (2.0 * ((double)L + LAMBDA));
  const double W_c_i = W_m_i;
  //const double W_m_t = 2.0 * (double)L * W_m_i + W_m_0; // total mean weight
  const double W_c_t = 2.0 * (double)L * W_c_i + W_c_0; // total cov weight

  /* calculate Cholesky decomposition of covariance matrix */
  int err;  
  aux::matrix tmp(x.getCovariance());
  err = lapack::potrf('L', tmp);
  assert(err == 0);
  ublas::triangular_adaptor<aux::matrix, ublas::lower> cholesky(tmp);

  /* calculate sigma points 1,...,2L, propagate and sum as we go */
  unsigned int i, j;
  const aux::vector &x_mu = x.getExpectation();
  aux::matrix A(sqrt((double)L + LAMBDA) * cholesky);

  /* dimensionality of output unknown, so initialise vectors using sigma point
   * 1 */
  aux::vector X(x_mu + ublas::column(A,0));
  aux::vector X_mu_m(X);
  aux::vector Y(model.propagate(X,delta));
  aux::vector Y_mu_m(Y);
  aux::symmetric_matrix Y_sigma(outer_prod(Y,Y));
  aux::matrix* XY_sigma = P;
  if (XY_sigma != NULL) {
    *XY_sigma = outer_prod(X,Y);
  }

  /* include sigma point L+1 */
  noalias(X) = x_mu - ublas::column(A,0);
  noalias(X_mu_m) += X;
  noalias(Y) = model.propagate(X,delta);
  noalias(Y_mu_m) += Y;
  noalias(Y_sigma) += outer_prod(Y,Y);
  if (XY_sigma != NULL) {
    noalias(*XY_sigma) += outer_prod(X,Y);
  }

  /* include sigma points 2,...,L,L+2,...,2L */
  for (i = 2; i <= L; i++) {
    /* sigma point i */
    noalias(X) = x_mu + ublas::column(A,i-1);
    noalias(X_mu_m) += X;
    noalias(Y) = model.propagate(X,delta);
    noalias(Y_mu_m) += Y;
    noalias(Y_sigma) += outer_prod(Y,Y);
    if (XY_sigma != NULL) {
      noalias(*XY_sigma) += outer_prod(X,Y);
    }

    /* sigma point L+i */
    noalias(X) = x_mu - ublas::column(A,i-1);
    noalias(X_mu_m) += X;
    noalias(Y) = model.propagate(X,delta);
    noalias(Y_mu_m) += Y;
    noalias(Y_sigma) += outer_prod(Y,Y);
    if (XY_sigma != NULL) {
      noalias(*XY_sigma) += outer_prod(X,Y);
    }
  }

  aux::vector X_mu_c(X_mu_m*W_c_i);
  aux::vector Y_mu_c(Y_mu_m*W_c_i);
  X_mu_m *= W_m_i;
  Y_mu_m *= W_m_i;

  /* include sigma point 0 */
  noalias(X) = x_mu;
  noalias(X_mu_m) += W_m_0*X;
  noalias(X_mu_c) += W_c_0*X;
  noalias(Y) = model.propagate(X,delta);
  noalias(Y_mu_m) += W_m_0*Y;
  noalias(Y_mu_c) += W_c_0*Y;

  /* finalise covariance */
  Y_sigma *= W_c_i;
  noalias(Y_sigma) += W_c_0*outer_prod(Y,Y);
  noalias(Y_sigma) += W_c_t*outer_prod(Y_mu_m,Y_mu_m);
  
  //noalias(Y_sigma) -= outer_prod(Y_mu_m,Y_mu_c) + outer_prod(Y_mu_c,Y_mu_m);
  aux::matrix tmp1(outer_prod(Y_mu_m,Y_mu_c));
  noalias(Y_sigma) -= tmp1 + trans(tmp1);

  /* finalise cross-correlation */
  if (XY_sigma != NULL) {
    *XY_sigma *= W_c_i;
    noalias(*XY_sigma) += W_c_0*outer_prod(X,Y);
    noalias(*XY_sigma) += W_c_t*outer_prod(X_mu_m,Y_mu_m);
    noalias(*XY_sigma) -= outer_prod(X_mu_m,Y_mu_c)+outer_prod(X_mu_c,Y_mu_m);
  }

  /* create distribution */
  return aux::GaussianPdf(Y_mu_m, Y_sigma);
}

#endif
