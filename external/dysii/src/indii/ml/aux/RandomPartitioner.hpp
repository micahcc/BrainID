#ifndef INDII_ML_AUX_RANDOMPARTITIONER_HPP
#define INDII_ML_AUX_RANDOMPARTITIONER_HPP

#include "Partitioner.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Partitions a set of weighted samples into two sets along the midpoint of
 * a random dimension.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 477 $
 * @date $Date: 2008-07-24 23:39:18 +0100 (Thu, 24 Jul 2008) $
 */
class RandomPartitioner : public Partitioner {
public:
  /**
   * Destructor.
   */
  virtual ~RandomPartitioner();

  virtual bool init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is);
  
  virtual Partitioner::Partition assign(const aux::vector& x);

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
    indii::ml::aux::RandomPartitioner::assign(const aux::vector& x) {
  if (x(index) <= value) {
    return Partitioner::LEFT;
  } else {
    return Partitioner::RIGHT;
  }
}

#endif

