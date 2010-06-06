#ifndef INDII_ML_AUX_PARTITIONTREE_HPP
#define INDII_ML_AUX_PARTITIONTREE_HPP

#include "PartitionTreeNode.hpp"
#include "DiracMixturePdf.hpp"

#include "boost/serialization/split_member.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Abstract spatial partition tree.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 562 $
 * @date $Date: 2008-09-11 17:37:07 +0100 (Thu, 11 Sep 2008) $
 *
 * @section PartitionTree_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class PartitionTree {
public:
  /**
   * Default constructor.
   *
   * This should generally only be used when the object is to be
   * restored from a serialization.
   */
  PartitionTree();

  /**
   * Constructor.
   *
   * @param p Weighted sample set from which to build the tree.
   */
  PartitionTree(DiracMixturePdf* p);

  /**
   * Destructor.
   */
  virtual ~PartitionTree();

  /**
   * Clone tree.
   *
   * @return Clone of tree. Caller has ownership.
   */
  virtual PartitionTree* clone() = 0;

  /**
   * Get the underlying weighted sample set.
   *
   * @return The underlying weighted sample set.
   */
  DiracMixturePdf* getData();

  /**
   * Get the root node of the partition tree.
   *
   * @return Root node of the partition tree.
   */
  virtual PartitionTreeNode* getRoot() = 0;

  /**
   * Set the underlying weighted sample set.
   *
   * @param p The underlying weighted sample set.
   *
   * The new set should have the same number of components as the existing
   * set.
   */
  void setData(DiracMixturePdf* p);

  /**
   * Set the root node of the partition tree.
   *
   * @param root Root node of the partition tree.
   *
   * Care should be taken that the index of the greatest component
   * in the subtree @p root is not greater than the number of components
   * in the weighted sample set underlying the tree. This is not checked.
   */
  virtual void setRoot(PartitionTreeNode* root) = 0;

private:
  /**
   * Weighted sample set.
   */
  DiracMixturePdf* p;

  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned int version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned int version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;

};

    }
  }
}

inline indii::ml::aux::DiracMixturePdf*
    indii::ml::aux::PartitionTree::getData() {
  return p;
}

template<class Archive>
void indii::ml::aux::PartitionTree::save(Archive& ar,
    const unsigned int version) const {
  //
}

template<class Archive>
void indii::ml::aux::PartitionTree::load(Archive& ar,
    const unsigned int version) {
  //
}

#endif

