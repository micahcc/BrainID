#ifndef INDII_ML_AUX_DISTRIBUTEDPARTITIONER_HPP
#define INDII_ML_AUX_DISTRIBUTEDPARTITIONER_HPP

#include "Partitioner.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Partitions a distributed set of weighted points into two sets at the
 * \f$n\f$th component.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 489 $
 * @date $Date: 2008-07-31 12:13:05 +0100 (Thu, 31 Jul 2008) $
 *
 * The purpose of this partitioner is to evenly distribute components across
 * nodes for DiracMixturePdf::redistributeBySpace(). It is for internal
 * use only.
 */
class DistributedPartitioner : public Partitioner {
public:
  /**
   * Constructor.
   *
   * @param nth No. components to be on left side of split.
   */
  DistributedPartitioner(const unsigned int nth);

  /**
   * Destructor.
   */
  virtual ~DistributedPartitioner();

  virtual bool init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is);
  
  virtual Partitioner::Partition assign(const vector& x);

private:
  /**
   * No. components to be on left side of split.
   */
  unsigned int nth;

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
    indii::ml::aux::DistributedPartitioner::assign(const vector& x) {
  if (x(index) < value) { // <, not <=, important given how vlaue is selected
    return Partitioner::LEFT;
  } else {
    return Partitioner::RIGHT;
  }
}

#endif

