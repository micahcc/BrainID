#include "PartitionTree.hpp"

#include <stack>

using namespace indii::ml::aux;

PartitionTree::PartitionTree() : p(NULL) {
  //
}

PartitionTree::PartitionTree(DiracMixturePdf* p) : p(p) {
  //
}

PartitionTree::~PartitionTree() {
  //
}

void PartitionTree::setData(DiracMixturePdf* p) {
  /* pre-condition */
  assert (this->p->getSize() == p->getSize());
  
  this->p = p;
}

