#include "PartitionFunctor.hpp"

using namespace indii::ml::aux;

PartitionFunctor::PartitionFunctor(const DiracMixturePdf& p,
    const unsigned int index, const double value) : p(p), index(index),
    value(value) {
  //
}

PartitionFunctor::~PartitionFunctor() {
  //
}

