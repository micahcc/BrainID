#ifndef INDII_ML_AUX_VARIANCEPARTITIONER_HPP
#define INDII_ML_AUX_VARIANCEPARTITIONER_HPP

#include "Partitioner.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Partitions a set of weighted points into two sets at the mean of the
 * dimension with greatest variance.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 477 $
 * @date $Date: 2008-07-24 23:39:18 +0100 (Thu, 24 Jul 2008) $
 *
 * @bug Points with negligible but nonzero weight may not be able to be 
 * split off from others using the mean. Consider two points, one with
 * negligible weight, the other which consequently equals the mean. Both
 * points may therefore lie on the same side of the split.
 */
class VariancePartitioner : public Partitioner {
public:
  /**
   * Destructor.
   */
  virtual ~VariancePartitioner();

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
    indii::ml::aux::VariancePartitioner::assign(const aux::vector& x) {
  /* pre-condition */
  assert (x.size() > index);

  if (x(index) <= value) {
    return Partitioner::LEFT;
  } else {
    return Partitioner::RIGHT;
  }
}

#endif

