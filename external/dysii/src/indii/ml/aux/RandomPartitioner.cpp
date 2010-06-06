#include "RandomPartitioner.hpp"

using namespace indii::ml::aux;

RandomPartitioner::~RandomPartitioner() {
  //
}

bool RandomPartitioner::init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is) {
  /* pre-condition */
  assert (is.size() >= 2);

  unsigned int i;
  double lower, upper, x;

  /* randomly select dimension */
  index = static_cast<unsigned int>(Random::uniform(0, p->getDimensions()));

  /* split on midpoint of selected dimension */
  lower = p->get(is[0])(index);
  upper = lower;
  for (i = 1; i < is.size(); i++) {
    x = p->get(is[i])(index);
    if (x < lower) {
      lower = x;
    } else if (x > upper) {
      upper = x;
    }
  }
  this->value = (lower + upper) / 2.0;
  
  return (upper - lower > 0.0);
}

