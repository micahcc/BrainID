#ifndef INDII_ML_AUX_PARTITIONFUNCTOR_HPP
#define INDII_ML_AUX_PARTITIONFUNCTOR_HPP

#include "DiracMixturePdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Predicate functor for nth element partition. 
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 489 $
 * @date $Date: 2008-07-31 12:13:05 +0100 (Thu, 31 Jul 2008) $
 *
 * Internal use only, by DistributedPartitioner.
 */
class PartitionFunctor {
public:
  /**
   * Constructor.
   *
   * @param p Weighted sample set.
   * @param index Dimension on which to split.
   * @param value Value on which to split.
   */
  PartitionFunctor(const DiracMixturePdf& p, const unsigned int index, 
      const double value);

  /**
   * Destructor.
   */
  virtual ~PartitionFunctor();
      
  /**
   * Apply function.
   *
   * @param i Index of the component in the weighted sample set.
   *
   * @return True if dimension @p index of component @p i is then than
   * @p value.
   */
  bool operator()(const unsigned int i);
  
private:
  /**
   * The weighted sample set.
   */
  const DiracMixturePdf& p;
  
  /**
   * Dimension on which to split.
   */
  const unsigned int index;
  
  /**
   * Value on which to split.
   */
  const double value;
  
};

    }
  }
}

inline bool indii::ml::aux::PartitionFunctor::operator()(
    const unsigned int i) {
  return p.get(i)[index] < value;
}

#endif

