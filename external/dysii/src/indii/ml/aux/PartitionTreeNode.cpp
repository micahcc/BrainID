#include "PartitionTreeNode.hpp"

using namespace indii::ml::aux;

PartitionTreeNode::PartitionTreeNode() : size(0), left(NULL), right(NULL) {
  //
}

PartitionTreeNode::PartitionTreeNode(const unsigned int i,
    const unsigned int depth) : flagLeaf(true), flagPrune(false),
    flagInternal(false), depth(depth), size(1), i(i), is(0), left(NULL),
    right(NULL) {
  //
}

PartitionTreeNode::PartitionTreeNode(const std::vector<unsigned int>& is,
    const unsigned int depth) : flagLeaf(false), flagPrune(true),
    flagInternal(false), depth(depth), size(is.size()), i(0), is(is),
    left(NULL), right(NULL) {
  /* note that prune nodes with zero components are permitted, and indeed
   * required for DiracMixturePdf::redistributeBySpace() */ 
}     

PartitionTreeNode::PartitionTreeNode(PartitionTreeNode* left,
    PartitionTreeNode* right, const unsigned int depth) : flagLeaf(false),
    flagPrune(false), flagInternal(true), depth(depth),
    size(left->getSize() + right->getSize()), i(0), is(0), left(left),
    right(right) {
  //
}

PartitionTreeNode::PartitionTreeNode(const PartitionTreeNode& o) {
  flagLeaf = o.flagLeaf;
  flagPrune = o.flagPrune;
  flagInternal = o.flagInternal;
  depth = o.depth;
  size = o.size;
  i = o.i;
  is = o.is;
  
  if (o.left == NULL) {
    left = NULL;
  } else {
    left = o.left->clone();
  }
  
  if (o.right == NULL) {
    right = NULL;
  } else {
    right = o.right->clone();
  }
}

PartitionTreeNode::~PartitionTreeNode() {
  delete left;
  delete right;
}

PartitionTreeNode& PartitionTreeNode::operator=(const PartitionTreeNode& o) {
  flagLeaf = o.flagLeaf;
  flagPrune = o.flagPrune;
  flagInternal = o.flagInternal;
  depth = o.depth;
  size = o.size;
  i = o.i;
  is = o.is;
  
  delete left;
  delete right;
  
  if (o.left == NULL) {
    left = NULL;
  } else {
    left = o.left->clone();
  }
  
  if (o.right == NULL) {
    right = NULL;
  } else {
    right = o.right->clone();
  }
  
  return *this;
}

