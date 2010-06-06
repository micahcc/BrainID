#ifndef INDII_ML_AUX_ALMOST2NORM_HPP
#define INDII_ML_AUX_ALMOST2NORM_HPP

#include "Norm.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Vector 2-norm without square root, i.e. inner product.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 489 $
 * @date $Date: 2008-07-31 12:13:05 +0100 (Thu, 31 Jul 2008) $
 *
 * Almost2Norm is not strictly a norm, as it does not satisfy the property
 * of scalar multiplication. Combining with AlmostGaussianKernel, however,
 * produces the same result as using PNorm<2> and GaussianKernel, but is
 * much more efficient, as the square root in the norm and square in the 
 * exponent of the Gaussian are cancelled.
 *
 * @see AlmostGaussianKernel
 */
class Almost2Norm : public Norm {
public:
  /**
   * Destructor.
   */
  virtual ~Almost2Norm();

  virtual double operator()(const vector& x) const;

  virtual vector sample(const unsigned int N) const;

private:
  /**
   * Serialize.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
    
};

    }
  }
}

#include "Random.hpp"
#include "vector.hpp"

#include <set>

using namespace indii::ml::aux;

inline double indii::ml::aux::Almost2Norm::operator()(const vector& x)
    const {
  return inner_prod(x,x);
}

inline indii::ml::aux::vector indii::ml::aux::Almost2Norm::sample(
    const unsigned int N) const {   
  return Random::unitVector(N);
}

template<class Archive>
void indii::ml::aux::Almost2Norm::serialize(Archive& ar,
    const unsigned int version) {  
  ar & boost::serialization::base_object<Norm>(*this);
}

#endif

