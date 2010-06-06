#include "VariancePartitioner.hpp"

using namespace indii::ml::aux;

VariancePartitioner::~VariancePartitioner() {
  //
}

bool VariancePartitioner::init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is) {
  /* pre-condition */
  assert (is.size() >= 2);
    
  unsigned int i;
  double W = p->getTotalWeight();
  vector& first = p->get(is[0]);
  double w = p->getWeight(is[0]);

  vector mu(w * first);
  vector sigma(w * element_prod(first, first));

  for (i = 1; i < is.size(); i++) {
    vector& x(p->get(is[i]));
    w = p->getWeight(is[i]);
    
    noalias(mu) += w * x;
    noalias(sigma) += w * element_prod(x,x);
  }

  noalias(sigma) -= element_prod(mu, mu);
  mu /= W;
  sigma /= W;

  /* select dimension of highest variance */
  this->index = index_norm_inf(sigma);
  
  /* split on mean of given dimension */
  this->value = mu(index);
  
  return sigma(this->index) > 0.0;
}

