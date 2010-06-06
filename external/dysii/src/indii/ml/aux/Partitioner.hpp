#ifndef INDII_ML_AUX_PARTITIONER_HPP
#define INDII_ML_AUX_PARTITIONER_HPP

#include "DiracMixturePdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Partitions a set of weighted points into two sets for constructing a
 * partition tree.    
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 541 $
 * @date $Date: 2008-08-31 14:42:13 +0100 (Sun, 31 Aug 2008) $
 */
class Partitioner {
public:
  /**
   * Partitions.
   */
  enum Partition {
    LEFT,
    RIGHT
  };

  /**
   * Destructor.
   */
  virtual ~Partitioner();

  /**
   * Initialise the partitioner.
   *
   * @param p Weighted sample set.
   * @param is Indices of components of interest in the weighted sample
   * set.
   *
   * @return True if the partitioner is successful in finding a partition
   * point, false otherwise. The partitioner may be unsuccessful if, e.g.,
   * all points are identical or one point in a pair has negligible small
   * or zero weight.
   *
   * Initialises the partitioner after construction, optionally using
   * the given weighted sample set as a basis for the partition (e.g.
   * using its bounds or covariance).
   */
  virtual bool init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is) = 0;
  
  /**
   * Assign a sample to a partition.
   *
   * @param x The sample to assign.
   *
   * @return The partition to which the sample is assigned.
   */
  virtual Partition assign(const vector& x) = 0;

};
 
    }
  }
}

#endif

