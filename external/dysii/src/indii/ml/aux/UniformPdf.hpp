#ifndef INDII_ML_AUX_UNIFORMTREEPDF_HPP
#define INDII_ML_AUX_UNIFORMTREEPDF_HPP

#include "Pdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Uniform distribution over a hyper-rectangle.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 538 $
 * @date $Date: 2008-08-31 14:41:10 +0100 (Sun, 31 Aug 2008) $
 * 
 * @section UniformPdf_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
class UniformPdf : public Pdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the distribution with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  UniformPdf();

  /**
   * Constructor.
   *
   * @param lower Lower corner of the hyper-rectangle under the
   * distribution.
   * @param upper Upper corner of the hyper-rectangle under the
   * distribution.
   */
  UniformPdf(const vector& lower, const vector& upper);

  /**
   * Destructor.
   */
  ~UniformPdf();

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  virtual UniformPdf& operator=(const UniformPdf& o);

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);
  
  virtual const vector& getExpectation();

  virtual const symmetric_matrix& getCovariance();

  virtual vector sample();

  virtual double densityAt(const vector& x);

private:
  /**
   * Density of the distribution.
   */
  double p;

  /**
   * Lower corner of the hyper-rectangle under the distribution.
   */
  vector lower;
  
  /**
   * Upper corner of the hyper-rectangle under the distribution.
   */
  vector upper;

  /**
   * \f$\mathbf{\mu}\f$; mean of the distribution.
   */
  vector mu;

  /**
   * \f$\Sigma\f$; covariance of the distribution.
   */
  symmetric_matrix sigma;

  /**
   * Has \f$\mathbf{\mu}\f$ been calculated?
   */
  bool haveMu;
  
  /**
   * Has \f$\Sigma\f$ been calculated?
   */
  bool haveSigma;

  /**
   * Calculate \f$\mathbf{\mu}\f$.
   */
  void calculateExpectation();
  
  /**
   * Calculate \f$\Sigma\f$.
   */
  void calculateCovariance();

  /**
   * Calculate density over the hyper-rectangle defined by lower and upper.
   */
  void calculateDensity();

  /**
   * Serialize.
   */
  template<class Archive>
  void save(Archive& ar, const unsigned int version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned int version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;

};

    }
  }
}

template<class Archive>
void indii::ml::aux::UniformPdf::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<Pdf>(*this);
  ar & p;
  ar & lower;
  ar & upper;
}

template<class Archive>
void indii::ml::aux::UniformPdf::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<Pdf>(*this);
  ar & p;
  ar & lower;
  ar & upper;
  
  haveMu = false;
  haveSigma = false;
  mu.resize(N);
  sigma.resize(N);
}

#endif

