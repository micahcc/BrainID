#ifndef INDII_ML_AUX_NORM_HPP
#define INDII_ML_AUX_NORM_HPP

#include "vector.hpp"

#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Vector norm.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 420 $
 * @date $Date: 2008-04-03 20:56:55 +0100 (Thu, 03 Apr 2008) $
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class Norm {
public:
  /**
   * Destructor.
   */
  virtual ~Norm();

  /**
   * Evaluate the norm.
   *
   * @param x \f$\mathbf{x}\f$; a vector.
   *
   * @return \f$\|\mathbf{x}\|\f$; the norm of the vector.
   */
  virtual double operator()(const vector& x) const = 0;
  
  /**
   * Generate a random unit vector from a uniform distribution over the
   * unit vectors in the normed vector space.
   *
   * @param N Dimensionality of the normed vector space.
   *
   * @return Random unit vector of length N.
   */
  virtual vector sample(const unsigned int N) const = 0;

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

template<class Archive>
void indii::ml::aux::Norm::serialize(Archive& ar,
    const unsigned int version) {  
  //
}

#endif

