#include "MedianPartitioner.hpp"

using namespace indii::ml::aux;

MedianPartitioner::~MedianPartitioner() {
  //
}

bool MedianPartitioner::init(DiracMixturePdf* p,
      const std::vector<unsigned int>& is) {
  /* pre-condition */
  assert (is.size() >= 2);

  unsigned int i, j;
  vector lower(p->get(is[0]));
  vector upper(p->get(is[0]));
  vector length(p->getDimensions());
  
  /* calculate bounds */
  for (i = 1; i < is.size(); i++) {
    for (j = 0; j < p->getDimensions(); j++) {
      vector& x = p->get(is[i]);
      if (x(j) < lower(j)) {
        lower(j) = x(j);
      } else if (x(j) > upper(j)) {
        upper(j) = x(j);
      }
    }
  }

  /* select longest dimension */
  noalias(length) = upper - lower;
  this->index = index_norm_inf(length);   

  /* split on median of this dimension */
  std::vector<double> projection(is.size());
  unsigned int median = projection.size() / 2;

  for (i = 0; i < is.size(); i++) {
    projection[i] = p->get(is[i])(this->index);
  }
  std::nth_element(projection.begin(), projection.begin() + median,
      projection.end());
  this->value = projection[median];

  return length(this->index) > 0.0;
}

