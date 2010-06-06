#ifndef INDII_ML_AUX_GAUSSIANKERNEL_HPP
#define INDII_ML_AUX_GAUSSIANKERNEL_HPP

#include "Kernel.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Gaussian kernel for density estimation.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 531 $
 * @date $Date: 2008-08-25 16:09:23 +0100 (Mon, 25 Aug 2008) $
 *
 * The Gaussian kernel takes the form:
 *
 * \f[
 *   K(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}
 * \f]
 *
 * @see #hopt for guidance as to bandwidth selection.
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class GaussianKernel : public Kernel {
public:
  /**
   * Default constructor.
   *
   * This should generally only be used when the object is to be restored
   * from a serialization.
   */
  GaussianKernel();

  /**
   * Constructor.
   *
   * @param N \f$N\f$; dimensionality of the problem.
   * @param h \f$h\f$; the scaling parameter (bandwidth).
   *
   * Although the kernel itself is not intrinsically dependent on \f$N\f$
   * and \f$h\f$, its normalisation is. Supplying these allows substantial
   * performance increases through precalculations.
   */
  GaussianKernel(const unsigned int N, const double h);

  /**
   * Destructor.
   */
  virtual ~GaussianKernel();

  virtual double operator()(const double x) const;

  /**
   * Sample from the kernel.
   *
   * @return A sample from the kernel.
   */
  virtual double sample() const;
  
private:
  /**
   * \f$(h\sqrt{2\pi})^{-1}\f$; the normalisation term.
   */
  double ZI;
  
  /**
   * \f$(-2h^2)^{-1}\f$; the exponent term.
   */
  double E;

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

#include <math.h>

inline double indii::ml::aux::GaussianKernel::operator()(const double x)
    const {
  return ZI * exp(E * pow(x,2));
}

inline double indii::ml::aux::GaussianKernel::sample() const {
  return fabs(Random::gaussian(0.0, getBandwidth()));
}

template<class Archive>
void indii::ml::aux::GaussianKernel::serialize(Archive& ar,
    const unsigned int version) {  
  ar & boost::serialization::base_object<Kernel>(*this);
  ar & ZI;
  ar & E;
}

#endif

