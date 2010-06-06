#ifndef INDII_ML_FILTER_UNSCENTEDTRANSFORMATIONMODEL_HPP
#define INDII_ML_FILTER_UNSCENTEDTRANSFORMATIONMODEL_HPP

#include "../aux/vector.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * UnscentedTransformation compatible model. Represents the function
 * \f$f\f$ through which the Gaussian distributed random variable will
 * be propagated.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 301 $
 * @date $Date: 2007-09-10 23:56:50 +0100 (Mon, 10 Sep 2007) $
 */
template <class T = unsigned int>
class UnscentedTransformationModel {
public:
  /**
   * Destructor.
   */
  virtual ~UnscentedTransformationModel() = 0;

  /**
   * Propagate a sigma point \f$\mathcal{X}_i\f$ through the function \f$f\f$.
   *
   * @param X \f$\mathcal{X}_i\f$; the sigma point.
   * @param delta \f$\Delta t\f$; length of time through which to propagate
   * the sigma point, if relevant.
   *
   * @return \f$\mathcal{Y}_i = f(\mathcal{X}_i,\Delta t)\f$
   */
  virtual indii::ml::aux::vector propagate(const indii::ml::aux::vector& X,
      T delta = 0) = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::UnscentedTransformationModel<T>::~UnscentedTransformationModel() {
  //
}

#endif
