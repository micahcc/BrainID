#ifndef INDII_ML_FILTER_RAUCHTUNGSTRIEBELSMOOTHERMODEL_HPP
#define INDII_ML_FILTER_RAUCHTUNGSTRIEBELSMOOTHERMODEL_HPP

#include "KalmanFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * RauchTungStriebelSmoother compatible model.
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
class RauchTungStriebelSmootherModel : virtual public KalmanFilterModel<T> {
public:
  /**
   * Destructor.
   */
  virtual ~RauchTungStriebelSmootherModel() = 0;

  /**
   * Perform smoothing update.
   *
   * @param p_xtnp1_ytT \f$P\big(\mathbf{x}(t_{n+1})\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_T)\big)\f$;
   * @param p_xtnp1_ytn \f$P\big(\mathbf{x}(t_{n+1})\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n)\big)\f$
   * @param p_xtn_ytn \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n)\big)\f$; @param delta
   * \f$t_{n+1} - t_n\f$;
   *
   * @return \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_T)\big)\f$
   */
  virtual indii::ml::aux::GaussianPdf p_xtn_ytT(
      const indii::ml::aux::GaussianPdf& p_xtnp1_ytT,
      const indii::ml::aux::GaussianPdf& p_xtnp1_ytn,
      const indii::ml::aux::GaussianPdf& p_xtn_ytn,
      const T delta) = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::RauchTungStriebelSmootherModel<T>::~RauchTungStriebelSmootherModel() {
  //
}

#endif
