#ifndef INDII_ML_FILTER_LINEARMODEL_HPP
#define INDII_ML_FILTER_LINEARMODEL_HPP

#include "../aux/matrix.hpp"
#include "../aux/vector.hpp"

#include "RauchTungStriebelSmootherModel.hpp"
#include "KalmanSmootherModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Simple linear model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 349 $
 * @date $Date: 2007-11-20 20:48:40 +0000 (Tue, 20 Nov 2007) $
 *
 * The system model takes the form:
 *
 * \f[\mathbf{x}_{t+1} = A\mathbf{x}_t + G\mathbf{w}_t\f]
 * 
 * where \f$\mathbf{w}_t\f$ is Gaussian noise with zero mean and
 * covariance matrix \f$Q\f$.
 *
 * The measurement model takes the form:
 *
 * \f[\mathbf{y}_t = C\mathbf{x}_t + \mathbf{v}_t\f]
 *
 * where \f$\mathbf{v}_t\f$ is Gaussian noise with zero mean and
 * covariance matrix \f$R\f$.
 *
 * For notational convenience, we define \f$\hat{\mathbf{x}}_{t|t}\f$ as
 * the expected value and \f$P_{t|t}\f$ as the covariance matrix of
 * the distribution \f$P\big(x_t\,|\,y_1,\ldots,y_t\big)\f$.
 */
class LinearModel : public RauchTungStriebelSmootherModel<unsigned int>,
    public KalmanSmootherModel<unsigned int> {
public:
  /**
   * Create new linear model.
   *
   * @param A \f$A\f$
   * @param G \f$G\f$
   * @param Q \f$Q\f$
   * @param C \f$C\f$
   * @param R \f$R\f$
   */
  LinearModel(indii::ml::aux::matrix& A, indii::ml::aux::matrix& G,
      indii::ml::aux::symmetric_matrix& Q, indii::ml::aux::matrix& C,
      indii::ml::aux::symmetric_matrix& R);

  /**
   * Destructor.
   */
  virtual ~LinearModel();

  /**
   * Predict next system state.
   *
   * \f{eqnarray*}
   * \hat{\mathbf{x}}_{t+1|t} & = & A\hat{\mathbf{x}}_{t|t} \\
   * P_{t+1|t} & = & AP_{t|t}A^T + GQG^T \\
   * \f}
   */
  virtual indii::ml::aux::GaussianPdf p_xtnp1_ytn(
      const indii::ml::aux::GaussianPdf& p_xtn_ytn, const unsigned int delta);

  /**
   * Refine prediction of next system state using next measurement.
   *
   * Let the Kalman gain be defined as:
   *
   * \f[K_{t+1} = P_{t+1|t}C^T(CP_{t+1|t}C^T + R)^{-1}\f]
   *
   * Then the measurement update proceeds as follows:
   *
   * \f{eqnarray*}
   * \hat{\mathbf{x}}_{t+1|t+1} & = & \hat{\mathbf{x}}_{t+1|t} +
   * K_{t+1}(\mathbf{y}_{t+1} - C\hat{\mathbf{x}}_{t+1|t}) \\
   * P_{t+1|t+1} & = & P_{t+1|t} - K_{t+1}CP_{t+1|t} \\
   * \f}
   */
  virtual indii::ml::aux::GaussianPdf p_xtnp1_ytnp1(
      const indii::ml::aux::GaussianPdf& p_xtnp1_ytn,
      const indii::ml::aux::vector& ytnp1, const unsigned int delta);

  /**
   * Predict measurement from system state.
   *
   * If p_x has mean \f$\hat{\mathbf{x}}\f$ and covariance \f$P_x\f$,
   * the return value has mean \f$\hat{\mathbf{y}}\f$ and covariance
   * \f$P_y\f$ defined by:
   *
   * \f{eqnarray*}
   * \hat{\mathbf{y}} & = & C\hat{\mathbf{x}} \\
   * P_y & = & C P_x C^T + R \\
   * \f}
   */
  virtual indii::ml::aux::GaussianPdf p_y_x(
      const indii::ml::aux::GaussianPdf& p_x);

  /**
   * Perform smoothing update.
   *
   * Let:
   *
   * \f[L_t = P_{t|t}A^TP_{t+1|t}^{-1}\f]
   *
   * The smoothing update proceeds as follows:
   *
   * \f{eqnarray*}
   * \hat{\mathbf{x}}_{t|T} & = & \hat{\mathbf{x}}_{t|t} +
   * L_t(\hat{\mathbf{x}}_{t+1|T} - \hat{\mathbf{x}}_{t+1|t}) \\
   * P_{t|T} & = & P_{t|t} + L_t(P_{t+1|T} - P_{t+1|t})L_t^T \\
   * \f}
   */
  virtual indii::ml::aux::GaussianPdf p_xtn_ytT(
      const indii::ml::aux::GaussianPdf& p_xtnp1_ytT,
      const indii::ml::aux::GaussianPdf& p_xtnp1_ytn,
      const indii::ml::aux::GaussianPdf& p_xtn_ytn,
      const unsigned int delta);

  virtual indii::ml::aux::GaussianPdf p_xtnm1_ytn(
      const indii::ml::aux::GaussianPdf& p_xtn_ytn, const unsigned int delta);

  virtual indii::ml::aux::GaussianPdf p_xtnm1_ytnm1(
      const indii::ml::aux::GaussianPdf& p_xtnm1_ytn,
      const indii::ml::aux::vector& ytnm1, const unsigned int delta);

private:
  /**
   * Size of the state space.
   */
  const unsigned int N;

  /**
   * Size of the measurement space.
   */
  const unsigned int M;

  /**
   * \f$A\f$
   */
  indii::ml::aux::matrix A;

  /**
   * \f$A^T\f$
   */
  indii::ml::aux::matrix AT;

  /**
   * \f$G\f$
   */
  indii::ml::aux::matrix G;

  /**
   * \f$G^T\f$
   */
  indii::ml::aux::matrix GT;

  /**
   * \f$Q\f$
   */
  indii::ml::aux::symmetric_matrix Q;

  /**
   * \f$C\f$
   */
  indii::ml::aux::matrix C;

  /**
   * \f$C^T\f$
   */
  indii::ml::aux::matrix CT;

  /**
   * \f$R\f$
   */
  indii::ml::aux::symmetric_matrix R;

  /**
   * \f$A^{-1}\f$ for backwards pass.
   */
  indii::ml::aux::matrix AI;

};

    }
  }
}

#endif

