#ifndef INDII_ML_AUX_DIRACPDF_HPP
#define INDII_ML_AUX_DIRACPDF_HPP

#include "Pdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {

/**
 * Dirac \f$\delta\f$-function probability density.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 570 $
 * @date $Date: 2008-09-16 16:49:33 +0100 (Tue, 16 Sep 2008) $
 *
 * Supports serialization through the Boost.Serialization library.
 */
class DiracPdf : public Pdf, public vector {
public:
  /**
   * Default constructor.
   *
   * Initialises the Dirac with zero dimensions. This should generally
   * only be used when the object is to be restored from a
   * serialization. Indeed, there is no other way to resize the Dirac
   * to nonzero dimensionality except by subsequently restoring from a
   * serialization.
   */
  DiracPdf();

  /**
   * Constructor.
   *
   * @param x The expectation of the Dirac.
   */
  DiracPdf(const vector& x);

  /**
   * Construct Dirac given dimensionality. The expectation remains
   * uninitialised, and should be set with setExpectation().
   *
   * @param N \f$N\f$; number of dimensions of the Dirac.
   */
  DiracPdf(unsigned int N);

  /**
   * Destructor.
   */
  virtual ~DiracPdf();

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  /**
   * Get the expected value of the Dirac.
   *
   * @return \f$\mathbf{\mu}\f$; expected value of the Dirac.
   */
  virtual const vector& getExpectation() const;

  /**
   * (Don't) get the covariance of the Dirac.
   *
   * @return \f$\mathbf{\Sigma}\f$; covariance of the Dirac.
   */
  virtual const symmetric_matrix& getCovariance() const;

  virtual const vector& getExpectation();

  virtual const symmetric_matrix& getCovariance();

  virtual vector sample();

  virtual double densityAt(const vector& x);

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

#include "boost/serialization/base_object.hpp"

inline const indii::ml::aux::vector&
    indii::ml::aux::DiracPdf::getExpectation() const {
  return *this;
}

inline const indii::ml::aux::vector&
    indii::ml::aux::DiracPdf::getExpectation() {
  return *this;
}

inline indii::ml::aux::vector indii::ml::aux::DiracPdf::sample() {
  return *this;
}

inline double indii::ml::aux::DiracPdf::densityAt(const vector& x) {
  if (norm_inf(*this - x) == 0.0) {
    return std::numeric_limits<double>::infinity();
  } else {
    return 0.0;
  }
}

template<class Archive>
void indii::ml::aux::DiracPdf::serialize(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<indii::ml::aux::Pdf>(*this);
  ar & boost::serialization::base_object<indii::ml::aux::vector>(*this);
}

#endif
