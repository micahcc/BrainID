#ifndef INDII_ML_FILTER_KERNELTWOFILTERSMOOTHERMODEL_HPP
#define INDII_ML_FILTER_KERNELTWOFILTERSMOOTHERMODEL_HPP

#include "ParticleFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * KernelTwoFilterSmoother compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 489 $
 * @date $Date: 2008-07-31 12:13:05 +0100 (Thu, 31 Jul 2008) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class KernelTwoFilterSmootherModel : public virtual ParticleFilterModel<T> {
public:
  /**
   * Destructor.
   */
  virtual ~KernelTwoFilterSmootherModel() = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::KernelTwoFilterSmootherModel<T>::~KernelTwoFilterSmootherModel() {
  //
}

#endif

