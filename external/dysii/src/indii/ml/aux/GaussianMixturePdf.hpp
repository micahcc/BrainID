#ifndef INDII_ML_AUX_GAUSSIANMIXTUREPDF_HPP
#define INDII_ML_AUX_GAUSSIANMIXTUREPDF_HPP

#include "StandardMixturePdf.hpp"
#include "GaussianPdf.hpp"
#include "DiracMixturePdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Gaussian mixture probability density.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 556 $
 * @date $Date: 2008-09-04 18:02:20 +0100 (Thu, 04 Sep 2008) $
 *
 * @see MixturePdf for more information regarding the serialization
 * and parallelisation features of this class.
 */
class GaussianMixturePdf : public StandardMixturePdf<GaussianPdf> {
public:
  /**
   * Default constructor.
   *
   * Initialises the mixture with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization. Indeed, there is no other way to resize the
   * mixture to nonzero dimensionality except by subsequently
   * restoring from a serialization.
   */
  GaussianMixturePdf();

  /**
   * Constructor. One or more components should be added with
   * addComponent() after construction.
   *
   * @param N Dimensionality of the distribution.
   */
  GaussianMixturePdf(const unsigned int N);

  /**
   * Constructor.
   *
   * @param K Number of Gaussian components.
   * @param p Weighted sample set.
   *
   * A @p K component Gaussian mixture is fit to @p p using Expectation-
   * Maximisation (EM) with random initialisation.
   *
   * @todo This has not been tested thoroughly.
   */
  GaussianMixturePdf(const unsigned int K, const DiracMixturePdf& p);

  /**
   * Destructor.
   */
  virtual ~GaussianMixturePdf();

private:
  /**
   * Serialize or restore from serialization.
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

template <class Archive>
void indii::ml::aux::GaussianMixturePdf::serialize(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<
      indii::ml::aux::StandardMixturePdf<indii::ml::aux::GaussianPdf> >(
      *this);
}

#endif

