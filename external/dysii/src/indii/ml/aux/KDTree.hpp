#ifndef INDII_ML_AUX_KDTREE_HPP
#define INDII_ML_AUX_KDTREE_HPP

#include "PartitionTree.hpp"
#include "KDTreeNode.hpp"
#include "MedianPartitioner.hpp"

#include "boost/serialization/split_member.hpp"

#include <vector>

namespace indii {
  namespace ml {
    namespace aux {
/**
 * \f$kd\f$ (k-dimensional) tree over a DiracMixturePdf.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 584 $
 * @date $Date: 2008-12-15 17:26:36 +0000 (Mon, 15 Dec 2008) $
 *
 * @param ST Space partitioner type, derived from Partitioner.
 *
 * @section KDTree_references References
 *
 * @anchor Gray2001
 * Gray, A. G. & Moore, A. W. `N-Body' Problems in Statistical
 * Learning. <i>Advances in Neural Information Processing Systems</i>,
 * <b>2001</b>, <i>13</i>.
 */
template <class ST = MedianPartitioner>
class KDTree : public PartitionTree {
public:
  /**
   * Default constructor.
   *
   * This should generally only be used when the object is to be
   * restored from a serialization.
   */
  KDTree();

  /**
   * Constructor.
   *
   * @param p Weighted sample set from which to build the tree.
   */
  KDTree(DiracMixturePdf* p);

  /**
   * Copy constructor.
   */
  KDTree(const KDTree<ST>& o);

  /**
   * Destructor.
   */
  virtual ~KDTree();

  virtual PartitionTree* clone();

  /**
   * Assignment operator.
   */
  KDTree<ST>& operator=(const KDTree<ST>& o);

  virtual PartitionTreeNode* getRoot();

  virtual void setRoot(PartitionTreeNode* root);

private:
  /**
   * Root node of the tree.
   */
  KDTreeNode* root;

  /**
   * Build \f$kd\f$ tree node.
   *
   * @param p Weighted sample set.
   * @param is Indices of the subset of components in p over which to build
   * the node.
   * @param depth Depth of the node in the tree. Zero for the root node.
   * 
   * @return The node. Caller has ownership.
   */
  static KDTreeNode* build(DiracMixturePdf* p,
      const std::vector<unsigned int>& is, const unsigned int depth = 0);

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

template<class ST>
indii::ml::aux::KDTree<ST>::KDTree() : root(NULL) {
  //
}

template<class ST>
indii::ml::aux::KDTree<ST>::KDTree(DiracMixturePdf* p) : PartitionTree(p) {
  unsigned int i;
  std::vector<unsigned int> is(p->getSize());

  for (i = 0; i < is.size(); i++) {
    is[i] = i;
  }

  if (is.size() > 0) {  
    root = build(p, is);
  } else {
    root = NULL;
  }
}

template<class ST>
indii::ml::aux::KDTree<ST>::KDTree(const KDTree<ST>& o) :
    PartitionTree(o) {
  if (o.root == NULL) {
    root = NULL;
  } else {
    root = dynamic_cast<KDTreeNode*>(o.root->clone());
  }
}

template<class ST>
indii::ml::aux::KDTree<ST>::~KDTree() {
  delete root;
}

template<class ST>
indii::ml::aux::KDTree<ST>& indii::ml::aux::KDTree<ST>::operator=(
    const KDTree<ST>& o) {
  PartitionTree::operator=(o);

  delete root;
  if (o.root == NULL) {
    root = NULL;
  } else {
    root = dynamic_cast<KDTreeNode*>(o.root->clone());
  }
  
  return *this;
}

template<class ST>
indii::ml::aux::PartitionTree* indii::ml::aux::KDTree<ST>::clone() {
  return new KDTree<ST>(*this);
}

template<class ST>
inline indii::ml::aux::PartitionTreeNode*
    indii::ml::aux::KDTree<ST>::getRoot() {
  return root;
}

template<class ST>
void indii::ml::aux::KDTree<ST>::setRoot(PartitionTreeNode* root) {
  this->root = dynamic_cast<KDTreeNode*>(root);
}

template<class ST>
indii::ml::aux::KDTreeNode* indii::ml::aux::KDTree<ST>::build(
    DiracMixturePdf* p, const std::vector<unsigned int>& is,
    const unsigned int depth) {
  /* pre-condition */
  assert (is.size() > 0);

  KDTreeNode* result;
  unsigned int i;
  
  if (is.size() == 1) {
    /* leaf node */
    result = new KDTreeNode(p, is.front(), depth);
  } else {
    /* internal node */
    ST partitioner;

    if (partitioner.init(p, is)) {
      std::vector<unsigned int> ls, rs; // indices of left, right components
      KDTreeNode *left, *right; // child nodes

      ls.reserve(is.size() / 2);
      rs.reserve(is.size() / 2);
      
      for (i = 0; i < is.size(); i++) {
        if (partitioner.assign(p->get(is[i])) == Partitioner::LEFT) {
          ls.push_back(is[i]);
        } else {
          rs.push_back(is[i]);
        }
      }

      if (ls.size() == 0) {
        /* degenerate case, prune node */
        result = new KDTreeNode(p, rs, depth);
      } else if (rs.size() == 0) {
        /* degenerate case, prune node */
        result = new KDTreeNode(p, ls, depth);        
      } else {
        /* internal node */
        left = build(p, ls, depth + 1);
        right = build(p, rs, depth + 1);
      
        result = new KDTreeNode(left, right, depth);
      }
    } else {
      /* Degenerate case, usually occurs when all points are identical or
         one has negligible weight, so that they cannot be partitioned
         spatially. Put them all into one prune node... */
      result = new KDTreeNode(p, is, depth);
    }
  }

  return result;
}

template<class ST>
template<class Archive>
void indii::ml::aux::KDTree<ST>::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<PartitionTree>(*this);

  const bool haveRoot = (root != NULL);
  ar & haveRoot;
  if (haveRoot) {
    ar & root;
  }
}

template<class ST>
template<class Archive>
void indii::ml::aux::KDTree<ST>::load(Archive& ar,
    const unsigned int version) {  
  bool haveRoot = false;
  
  if (root != NULL) {
    delete root;
    root = NULL;
  }
  
  ar & boost::serialization::base_object<PartitionTree>(*this);
  ar & haveRoot;
  if (haveRoot) {
    ar & root;
  }
}

#endif

