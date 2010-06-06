#ifndef INDII_ML_AUX_MEDIANPARTITIONER_HPP
#define INDII_ML_AUX_MEDIANPARTITIONER_HPP

#include "Partitioner.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Partitions a set of weighted points into two sets at the median of
 * the dimension with greatest range.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 489 $
 * @date $Date: 2008-07-31 12:13:05 +0100 (Thu, 31 Jul 2008) $
 */
class MedianPartitioner : public Partitioner {
public:
  /**
   * Destructor.
   */
  virtual ~MedianPartitioner();

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
    indii::ml::aux::MedianPartitioner::assign(const vector& x) {
  if (x(index) < value) { // <, not <=, important given how median is selected
    return Partitioner::LEFT;
  } else {
    return Partitioner::RIGHT;
  }
}

#endif

