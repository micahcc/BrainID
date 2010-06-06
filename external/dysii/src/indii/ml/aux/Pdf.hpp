#ifndef INDII_ML_AUX_PDF_HPP
#define INDII_ML_AUX_PDF_HPP

#include "vector.hpp"
#include "matrix.hpp"

namespace indii {
  namespace ml {
    namespace aux {

/**
 * Abstract probability distribution.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 538 $
 * @date $Date: 2008-08-31 14:41:10 +0100 (Sun, 31 Aug 2008) $
 */
class Pdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the density with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  Pdf();

  /**
   * Constructor.
   *
   * @param N \f$N\f$; number of dimensions of the density.
   */
  Pdf(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~Pdf();

  /**
   * Get the dimensionality of the distribution.
   *
   * @return Dimensionality of the distribution.
   */
  unsigned int getDimensions() const;

  /**
   * Set the dimensionality of the distribution.
   *
   * @param N Dimensionality of the distribution.
   * @param preserve True to preserve the current sufficient
   * statistics of the distribution in the lower dimensional space,
   * false if these may be discarded.
   */
  virtual void setDimensions(const unsigned int N,
      const bool preserve = false) = 0;

  /**
   * Get the expected value of the distribution.
   *
   * @return \f$\mathbf{\mu}\f$; expected value of the distribution.
   */
  virtual const vector& getExpectation() = 0;

  /**
   * Get the covariance of the distribution.
   *
   * @return \f$\Sigma\f$; covariance of the distribution.
   */
  virtual const symmetric_matrix& getCovariance() = 0;

  /**
   * Sample from the distribution.
   *
   * @return A sample from the distribution.
   */
  virtual vector sample() = 0;

  /**
   * Calculate the density at a given point.
   *
   * @param x \f$\mathbf{x}\f$; point at which to evaluate the
   * density.
   *
   * @return \f$p(\mathbf{x})\f$; the density at \f$\mathbf{x}\f$.
   */
  virtual double densityAt(const vector& x) = 0;

protected:
  /**
   * \f$N\f$; number of dimensions.
   */
  unsigned int N;

private:
  /**
   * Serialize, or restore from serialization.
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

inline unsigned int indii::ml::aux::Pdf::getDimensions() const {
  return N;
}

template<class Archive>
void indii::ml::aux::Pdf::serialize(Archive& ar, const unsigned int version) {
  ar & N;
}

#endif

