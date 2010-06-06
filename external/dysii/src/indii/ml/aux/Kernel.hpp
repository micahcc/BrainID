#ifndef INDII_ML_AUX_KERNEL_HPP
#define INDII_ML_AUX_KERNEL_HPP

#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * %Kernel for density estimation.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 477 $
 * @date $Date: 2008-07-24 23:39:18 +0100 (Thu, 24 Jul 2008) $
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class Kernel {
public:
  /**
   * Default constructor.
   *
   * This should generally only be used when the object is to be restored
   * from a serialization.
   */
  Kernel();

  /**
   * Constructor.
   *
   * @param h \f$h\f$; the scaling parameter (bandwidth).
   */
  Kernel(const double h);

  /**
   * Destructor.
   */
  virtual ~Kernel() = 0;

  /**
   * Get kernel bandwidth.
   *
   * @return Kernel bandwidth.
   */
  double getBandwidth() const;

  /**
   * Comparison operator for sorting etc.
   *
   * @param o Object to which to compare.
   *
   * @return True if the bandwidth of @p this is less than the bandwidth of
   * @p o.
   */
  bool operator<(const Kernel& o) const;

  /**
   * Evaluate the kernel.
   *
   * @param x Point at which to evaluate the kernel.
   *
   * @return Density of the kernel at the given point.
   */
  virtual double operator()(const double x) const = 0;
  
  virtual double sample() const = 0;

private:
  /**
   * \f$h\f$; the scaling parameter (bandwidth).
   */
  double h;

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

inline double indii::ml::aux::Kernel::getBandwidth() const {
  return h;
}

inline bool indii::ml::aux::Kernel::operator<(
    const indii::ml::aux::Kernel& o) const {
  return this->getBandwidth() < o.getBandwidth();
}

template<class Archive>
void indii::ml::aux::Kernel::serialize(Archive& ar,
    const unsigned int version) {  
  ar & h;
}

#endif

