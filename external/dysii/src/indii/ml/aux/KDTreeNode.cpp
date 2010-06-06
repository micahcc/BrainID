#include "KDTreeNode.hpp"

using namespace indii::ml::aux;

KDTreeNode::KDTreeNode() : lower(NULL), upper(NULL), ownLower(false),
    ownUpper(false) {
  //
}

KDTreeNode::KDTreeNode(DiracMixturePdf* p, const unsigned int i,
    const unsigned int depth) : PartitionTreeNode(i, depth) {  
  /* set bounds */
  vector* x = new vector(p->get(i));
  setLower(x, true);
  setUpper(x, false);
}

KDTreeNode::KDTreeNode(DiracMixturePdf* p,
    const std::vector<unsigned int>& is, const unsigned int depth) :
    PartitionTreeNode(is, depth) {    
  const unsigned int N = p->getDimensions();
  unsigned int i, j;
  
  if (is.size() > 0) { // mightn't be for DiracMixturePdf::distributedBuild()
    vector* lower = new vector(p->get(is[0]));
    vector* upper = new vector(p->get(is[0]));
  
    /* calculate bounds */
    for (i = 1; i < is.size(); i++) {
      vector& x = p->get(is[i]);
      for (j = 0; j < N; j++) {
        if (x(j) < (*lower)(j)) {
          (*lower)(j) = x(j);
        } else if (x(j) > (*upper)(j)) {
          (*upper)(j) = x(j);
        }
      }
    }  

    /* set bounds */
    setLower(lower, true);
    setUpper(upper, true);
  } else {
    setLower(NULL, false);
    setUpper(NULL, false);
  }
}

KDTreeNode::KDTreeNode(KDTreeNode* left, KDTreeNode* right,
    const unsigned int depth) : PartitionTreeNode(left, right, depth) {
  unsigned int i;

  /* calculate lower bound */
  vector* l1 = left->lower;
  vector* l2 = right->lower;
  vector* lower = NULL;
  bool ownLower = false;

  if (l1 == NULL) {
    if (l2 == NULL) {
      lower = NULL;
    } else {
      lower = l2;
    }
  } else if (l2 == NULL) {
    lower = l1;
  } else {
    assert (l1->size() == l2->size());
    lower = new vector(*l1);
    for (i = 0; i < l2->size(); i++) {
      if ((*l2)(i) < (*lower)(i)) {
        (*lower)(i) = (*l2)(i);
        ownLower = true;
      }
    }
    if (!ownLower) {
      /* lower bound same as left child, so just copy pointer */
      delete lower;
      lower = l1;
    }
  }
  
  /* calculate upper bound */
  vector* u1 = left->upper;
  vector* u2 = right->upper;
  vector* upper = NULL;
  bool ownUpper = false;

  if (u1 == NULL) {
    if (u2 == NULL) {
      upper = NULL;
    } else {
      upper = u2;
    }
  } else if (u2 == NULL) {
    upper = u1;
  } else {
    assert (u1->size() == u2->size());  
    upper = new vector(*u2);
    for (i = 0; i < u1->size(); i++) {
      if ((*u1)(i) > (*upper)(i)) {
        (*upper)(i) = (*u1)(i);
        ownUpper = true;
      }
    }
    if (!ownUpper) {
      /* upper bound same as right child, so just copy pointer */
      delete upper;
      upper = u2;
    }
  }
  
  /* set bounds */
  setLower(lower, ownLower);
  setUpper(upper, ownUpper);
}

KDTreeNode::KDTreeNode(const KDTreeNode& o) : PartitionTreeNode(o) {
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

  if (o.ownLower) {
    lower = new vector(*o.lower);
  } else {
    if (left->lower != NULL) {
      lower = left->lower;
    } else {
      lower = right->lower;
    }
  }
  if (o.ownUpper) {
    upper = new vector(*o.upper);
  } else {
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
  
  ownLower = o.ownLower;
  ownUpper = o.ownUpper;
}

KDTreeNode& KDTreeNode::operator=(const KDTreeNode& o) {
  PartitionTreeNode::operator=(o);

  if (ownLower) {
    delete lower;
    lower = NULL;
  }
  if (ownUpper) {
    delete upper;
    upper = NULL;
  }

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

  if (o.ownLower) {
    lower = new vector(*o.lower);
  } else {
    if (left->lower != NULL) {
      lower = left->lower;
    } else {
      lower = right->lower;
    }
  }
  if (o.ownUpper) {
    upper = new vector(*o.upper);
  } else {
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
  
  ownLower = o.ownLower;
  ownUpper = o.ownUpper;

  return *this;
}

KDTreeNode::~KDTreeNode() {
  if (ownLower) {
    delete lower;
    lower = NULL; // in case lower == upper
  }
  if (ownUpper) {
    delete upper;
    upper = NULL;
  }
}

PartitionTreeNode* KDTreeNode::clone() const {
  return new KDTreeNode(*this);
}

void KDTreeNode::setLower(vector* lower, const bool own) {
  this->lower = lower;
  this->ownLower = own;
}

void KDTreeNode::setUpper(vector* upper, const bool own) {
  this->upper = upper;
  this->ownUpper = own;
}

void KDTreeNode::difference(const vector& x, vector& result) const {
  /* pre-condition */
  assert (x.size() == getLower()->size());
  
  const unsigned int N = getLower()->size();

  if (isLeaf()) {
    noalias(result) = x - *this->getLower();
  } else {
    const vector& lower = *getLower();
    const vector& upper = *getUpper();
    unsigned int i;

    for (i = 0; i < N; i++) {
      if (x(i) < lower(i)) {
        result(i) = lower(i) - x(i);
      } else if (x(i) > upper(i)) {
        result(i) = x(i) - upper(i);
      } else {
        result(i) = 0;
      }
    }
  }
}

void KDTreeNode::difference(const PartitionTreeNode& node, vector& result)
    const {
  const KDTreeNode& kdNode = static_cast<const KDTreeNode&>(node);
  
  /* pre-conditions */
  assert (kdNode.getLower()->size() == getLower()->size());
  assert (kdNode.getLower()->size() == result.size());

  const unsigned int N = getLower()->size();

  if (isLeaf()) {
    kdNode.difference(*getLower(), result);
  } else {
    const vector& lower = *getLower();
    const vector& upper = *getUpper();
    const vector& olower = *kdNode.getLower();
    const vector& oupper = *kdNode.getUpper();
    unsigned int i;

    for (i = 0; i < N; i++) {
      if (oupper(i) < lower(i)) {
        result(i) = lower(i) - oupper(i);
      } else if (olower(i) > upper(i)) {
        result(i) = olower(i) - upper(i);
      } else {
        result(i) = 0;
      }
    }
  }  
}

