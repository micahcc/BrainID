#ifndef INDII_ML_AUX_PARTITIONTREENODE_HPP
#define INDII_ML_AUX_PARTITIONTREENODE_HPP

#include "vector.hpp"

#include "boost/serialization/split_member.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Node of a spatial partition tree.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 590 $
 * @date $Date: 2008-12-17 15:09:40 +0000 (Wed, 17 Dec 2008) $
 *
 * @section PartitionTreeNode_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class PartitionTreeNode {
public:
  /**
   * Constructor.
   *
   * This should generally only be used when the object is to be
   * restored from a serialization.
   */
  PartitionTreeNode();

  /**
   * Constructor for leaf node.
   *
   * @param i Index of component in the weighted sample set.
   * @param depth Depth of the node in that tree.
   */
  PartitionTreeNode(const unsigned int i, const unsigned int depth);
   
  /**
   * Constructor for prune node.
   *
   * @param is Indices of components in the weighted sample set.
   * @param depth Depth of the node in that tree.
   */
  PartitionTreeNode(const std::vector<unsigned int>& is,
      const unsigned int depth);
  
  /**
   * Constructor for internal node.
   *
   * @param left Left child node. Caller releases ownership.
   * @param right Right child node. Caller releases ownership.
   * @param depth Depth of the node in that tree.
   */
  PartitionTreeNode(PartitionTreeNode* left, PartitionTreeNode* right,
      const unsigned int depth);

  /**
   * Copy constructor.
   */
  PartitionTreeNode(const PartitionTreeNode& o);
  
  /**
   * Destructor.
   */
  virtual ~PartitionTreeNode();

  /**
   * Assignment operator.
   */
  PartitionTreeNode& operator=(const PartitionTreeNode& o);

  /**
   * Clone node.
   *
   * @return Clone of node. Caller has ownership.
   */
  virtual PartitionTreeNode* clone() const = 0;

  /**
   * Get the depth of the node in its tree.
   *
   * @return The depth of the node.
   */
  unsigned int getDepth() const;

  /**
   * Is the node a leaf node?
   *
   * @return True if the node is a leaf node, false otherwise.
   */
  bool isLeaf() const;
  
  /**
   * Is the node a pruned node?
   *
   * @return True if the node is a pruned node, false otherwise.
   */
  bool isPrune() const;
  
  /**
   * Is the node an internal node?
   *
   * @return True if the node is an internal node, false otherwise.
   */
  bool isInternal() const;

  /**
   * Get the number of components encompassed by the node.
   *
   * @return The number of components encompassed by the node.
   */
  unsigned int getSize() const;

  /**
   * Get the component index of a leaf node.
   *
   * @return The component index, if a leaf node.
   */
  unsigned int getIndex() const;

  /**
   * Get the component indices of a pruned node.
   *
   * @return The component indices, if a pruned node.
   */
  const std::vector<unsigned int>& getIndices() const;

  /**
   * Get the left child of the node.
   *
   * @return The left child of an internal node.
   */
  PartitionTreeNode* getLeft();
  
  /**
   * Get the right child of the node.
   *
   * @return The right child of an internal node.
   */
  PartitionTreeNode* getRight();

  /**
   * Find the coordinate difference of the node from a single point.
   *
   * @param x Query point.
   * @param result After return, difference between the query point and
   * the nearest point within the volume contained by the node.
   *
   * Note that the difference may contain negative values. Usually a norm
   * would subsequently be applied to obtain a scalar distance.
   */
  virtual void difference(const vector& x, vector& result) const = 0;

  /**
   * Find the coordinate difference of the node from another node.
   *
   * @param node Query node.
   * @param result After return, difference between the closest two points
   * in the volumes contained by the nodes.
   *
   * Note that the difference may contain negative values. Usually a norm
   * would subsequently be applied to obtain a scalar distance.
   */
  virtual void difference(const PartitionTreeNode& node, vector& result)
      const = 0;
  
private:
  /**
   * Is the node a leaf node?
   */
  bool flagLeaf;
  
  /**
   * Is the node a pruned node?
   */
  bool flagPrune;
  
  /**
   * Is the node an internal node?
   */
  bool flagInternal;

  /**
   * Depth of the node.
   */
  unsigned int depth;

  /**
   * Number of components encompassed by the node.
   */
  unsigned int size;

  /**
   * Component index for leaf node.
   */
  unsigned int i;

  /**
   * Component indices for prune node.
   */
  std::vector<unsigned int> is;

  /**
   * The left child for an internal node.
   */
  PartitionTreeNode* left;
  
  /**
   * The right child for an internal node.
   */
  PartitionTreeNode* right;

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

inline unsigned int indii::ml::aux::PartitionTreeNode::getIndex() const {
  /* pre-condition */
  assert (flagLeaf);
  
  return i;
}

inline const std::vector<unsigned int>&
    indii::ml::aux::PartitionTreeNode::getIndices() const {
  return is;
}

inline indii::ml::aux::PartitionTreeNode*
    indii::ml::aux::PartitionTreeNode::getLeft() { 
  return left;
}
  
inline indii::ml::aux::PartitionTreeNode*
    indii::ml::aux::PartitionTreeNode::getRight() {
  return right;
}

inline unsigned int indii::ml::aux::PartitionTreeNode::getDepth() const {
  return depth;
}

inline unsigned int indii::ml::aux::PartitionTreeNode::getSize() const {
  return size;
}

inline bool indii::ml::aux::PartitionTreeNode::isLeaf() const {
  return flagLeaf;
}

inline bool indii::ml::aux::PartitionTreeNode::isPrune() const {
  return flagPrune;
}

inline bool indii::ml::aux::PartitionTreeNode::isInternal() const {
  return flagInternal;
}

template<class Archive>
void indii::ml::aux::PartitionTreeNode::load(Archive& ar,
    const unsigned int version) {
  delete left;
  delete right;

  ar & flagLeaf;
  ar & flagPrune;
  ar & flagInternal;
  ar & depth;
  ar & size;
  ar & i;
  ar & is;
  if (flagInternal) {
    ar & left;
    ar & right;
  } else {
    left = NULL;
    right = NULL;
  }
}

template<class Archive>
void indii::ml::aux::PartitionTreeNode::save(Archive& ar,
    const unsigned int version) const {   
  ar & flagLeaf;
  ar & flagPrune;
  ar & flagInternal;
  ar & depth;
  ar & size;
  ar & i;
  ar & is;
  if (flagInternal) {
    ar & left;
    ar & right;
  }
}

#endif

