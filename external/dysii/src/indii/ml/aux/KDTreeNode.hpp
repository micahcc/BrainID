#ifndef INDII_ML_AUX_KDTREENODE_HPP
#define INDII_ML_AUX_KDTREENODE_HPP

#include "PartitionTreeNode.hpp"
#include "DiracMixturePdf.hpp"

#include "boost/serialization/split_member.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Node of a \f$kd\f$ tree.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 570 $
 * @date $Date: 2008-09-16 16:49:33 +0100 (Tue, 16 Sep 2008) $
 *
 * @section KDTreeNode_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class KDTreeNode : public PartitionTreeNode {
public:
  /**
   * Default constructor.
   *
   * This should generally only be used when the object is to be
   * restored from a serialization.
   */
  KDTreeNode();

  /**
   * Construct leaf node.
   *
   * @param p Weighted sample set.
   * @param i Index of component in the weighted sample set.
   * @param depth Depth of the node in the tree.
   */
  KDTreeNode(DiracMixturePdf* p, unsigned int i, unsigned int depth);

  /**
   * Construct prune node.
   *
   * @param p Weighted sample set.
   * @param is Indices of components in the weighted sample set.
   * @param depth Depth of the node in the tree.
   */
  KDTreeNode(DiracMixturePdf* p, const std::vector<unsigned int>& is,
      const unsigned int depth);

  /**
   * Construct internal node.
   *
   * @param left Left child node. Caller releases ownership.
   * @param right Right child node. Caller releases ownership.
   * @param depth Depth of the node in the tree.
   */
  KDTreeNode(KDTreeNode* left, KDTreeNode* right, const unsigned int depth);

  /**
   * Copy constructor.
   */
  KDTreeNode(const KDTreeNode& o);

  /**
   * Destructor.
   */
  virtual ~KDTreeNode();

  /**
   * Assignment operator.
   */
  KDTreeNode& operator=(const KDTreeNode& o);

  virtual PartitionTreeNode* clone() const;

  /**
   * Get lower bound on the node.
   */
  const vector* getLower() const;
  
  /**
   * Get upper bound on the node.
   */
  const vector* getUpper() const;

  virtual void difference(const vector& x, vector& result) const;

  virtual void difference(const PartitionTreeNode& node, vector& result)
      const;

private:  
  /**
   * The lower bound.
   */
  vector* lower;
  
  /**
   * The upper bound.
   */
  vector* upper;
  
  /**
   * Does object own the lower bound?
   */
  bool ownLower;
  
  /**
   * Does object own the upper bound?
   */
  bool ownUpper;

  /**
   * Set the lower bound on the node.
   *
   * @param lower Lower bound on the node.
   * @param own True if ownership of the object should be assumed.
   */
  void setLower(vector* lower, const bool own);
  
  /**
   * Set the upper bound on the node.
   *
   * @param upper Upper bound on the node.
   * @param own True if ownership of the object should be assumed.
   */
  void setUpper(vector* upper, const bool own);

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

#include "boost/serialization/base_object.hpp"

inline const indii::ml::aux::vector* indii::ml::aux::KDTreeNode::getLower()
    const {
  return lower;
}

inline const indii::ml::aux::vector* indii::ml::aux::KDTreeNode::getUpper()
    const {
  return upper;
}

template<class Archive>
void indii::ml::aux::KDTreeNode::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<PartitionTreeNode>(*this);

  ar & ownLower;
  if (ownLower) {
    ar & lower;
  }
  ar & ownUpper;
  if (ownUpper) {
    ar & upper;
  }
}

template<class Archive>
void indii::ml::aux::KDTreeNode::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<PartitionTreeNode>(*this);
    
  if (ownLower) {
    delete lower;
    lower = NULL;
  }
  if (ownUpper) {
    delete upper;
    upper = NULL;
  }
  
  ar & ownLower;
  if (ownLower) {
    ar & lower;
  }
  ar & ownUpper;
  if (ownUpper) {
    ar & upper;
  }

  /* tracking of indii::ml::aux::vector may be turned off, and commonly is,
     so restore non-owned bounds separately */
  KDTreeNode* left;
  KDTreeNode* right;
  if (getLeft() == NULL) {
    left = NULL;
  } else {
    left = dynamic_cast<KDTreeNode*>(getLeft());
  }
  if (getRight() == NULL) {
    right = NULL;
  } else {
    right = dynamic_cast<KDTreeNode*>(getRight());
  }

  if (!ownLower) {
    if (left->lower != NULL) {
      lower = left->lower;
    } else {
      lower = right->lower;
    }
  }
  if (!ownUpper) {
    if (isLeaf()) {
      upper = lower; // see constructor
    } else {
      if (right->upper != NULL) {
	upper = right->upper;
      } else {
	upper = left->upper;
      }
    }
  }
}

#endif

