#ifndef INDII_ML_FILTER_KERNELFORWARDBACKWARDSMOOTHERMODEL_HPP
#define INDII_ML_FILTER_KERNELFORWARDBACKWARDSMOOTHERMODEL_HPP

#include "ParticleFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * KernelForwardBackwardSmoother compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 490 $
 * @date $Date: 2008-07-31 17:20:07 +0100 (Thu, 31 Jul 2008) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class KernelForwardBackwardSmootherModel :
    public virtual ParticleFilterModel<T> {
public:
  /**
   * Destructor.
   */
  virtual ~KernelForwardBackwardSmootherModel() = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::KernelForwardBackwardSmootherModel<T>::~KernelForwardBackwardSmootherModel() {
  //
}

#endif

