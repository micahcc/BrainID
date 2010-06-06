#ifndef INDII_ML_AUX_PNORM_HPP
#define INDII_ML_AUX_PNORM_HPP

#include "Norm.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Vector p-norm, \f$\|\cdot\|_p\f$.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 420 $
 * @date $Date: 2008-04-03 20:56:55 +0100 (Thu, 03 Apr 2008) $
 *
 * @param P Degree of the p-norm.
 */
template <unsigned int P>
class PNorm : public Norm {
public:
  /**
   * Destructor.
   */
  virtual ~PNorm();

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

template<>
vector PNorm<2>::sample(const unsigned int N) const;

    }
  }
}

#include "Random.hpp"
#include "vector.hpp"

#include <set>

using namespace indii::ml::aux;

template<unsigned int P>
indii::ml::aux::PNorm<P>::~PNorm() {
  //
}

template<unsigned int P>
inline double indii::ml::aux::PNorm<P>::operator()(const vector& x)
    const {
  return norm<P,vector>(x);
}

template<>
indii::ml::aux::vector indii::ml::aux::PNorm<2>::sample(
    const unsigned int N) const {   
  return Random::unitVector(N);
}

template<unsigned int P>
indii::ml::aux::vector indii::ml::aux::PNorm<P>::sample(
    const unsigned int N) const {   
  unsigned int i;
  std::multiset<double> u;
  std::multiset<double>::iterator iter, end;
  vector x(N);
  
  /* sample independent uniform variates on [0,1] and sort */
  for (i = 0; i < N - 1; i++) {
    u.insert(Random::uniform(0.0, 1.0));
  }
  
  /* calculate increments */
  if (!u.empty()) {
    iter = u.begin();
    end = u.end();
    x(0) = *iter;
    iter++;
    for (i = 1; iter != end; iter++, i++) {
      x(i) = *iter - x(i - 1);
    }
    x(N - 1) = 1.0 - x(N - 2);
  } else {
    x(0) = 1.0;
  }
  //assert (sum(x) == 1.0);
  
  /* normalise to unit vector in given norm */
  for (i = 0; i < N; i++) {
    x(i) = pow(x(i), 1.0 / P);
    if (Random::bernoulli() == 0) {
      x(i) *= -1.0;
    }
  }
  
  return x;
  
  /* alternative angular coordinates approach (for 2-norm only)? See
   * Wikipedia article on "n-sphere" */
  //unsigned int i;
  //double y;
  //vector x(N), phi(N - 1);
  
  /* sample angular coordinates (radians), last has range of 2*PI,
   * others PI */
  //if (N >= 2) {
  //  for (i = 0; i < N - 2; i++) {
  //    phi(i) = Random::uniform(0.0, M_PI);
  //  }
  //  phi(N - 2) = Random::uniform(0.0, 2.0*M_PI);
  //}
  
  /* calculate cartesian coordinates */
  //y = 1.0;
  //for (i = 0; i < N - 1; i++) {
  //  x(i) = y * cos(phi(i));
  //  y *= sin(phi(i));
  //}
  //x(N - 1) = y;
}

template<unsigned int P>
template<class Archive>
void indii::ml::aux::PNorm<P>::serialize(Archive& ar,
    const unsigned int version) {  
  ar & boost::serialization::base_object<Norm>(*this);
}

#endif

