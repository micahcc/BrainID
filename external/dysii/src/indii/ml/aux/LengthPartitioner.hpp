#ifndef INDII_ML_AUX_LENGTHPARTITIONER_HPP
#define INDII_ML_AUX_LENGTHPARTITIONER_HPP

#include "Partitioner.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Partitions a set of weighted points into two sets at the midpoint of
 * the widest dimension.    
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 477 $
 * @date $Date: 2008-07-24 23:39:18 +0100 (Thu, 24 Jul 2008) $
 */
class LengthPartitioner : public Partitioner {
public:
  /**
   * Destructor.
   */
  virtual ~LengthPartitioner();

  virtual bool init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is);
  
  virtual Partitioner::Partition assign(const vector& x);

private:
  /**
   * Index of the dimension on which to split.
   */
  unsigned int index;
  
  /**
   * Value along which to split.
   */
  double value;

};
 
    }
  }
}

inline indii::ml::aux::Partitioner::Partition
    indii::ml::aux::LengthPartitioner::assign(const vector& x) {
  if (x(index) <= value) {
    return Partitioner::LEFT;
  } else {
    return Partitioner::RIGHT;
  }
}

#endif

