#ifndef INDII_ML_FILTER_PARTICLERESAMPLER_HPP
#define INDII_ML_FILTER_PARTICLERESAMPLER_HPP

#include "../aux/DiracMixturePdf.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Resampler for particle %filter.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 404 $
 * @date $Date: 2008-03-05 14:52:55 +0000 (Wed, 05 Mar 2008) $
 */
class ParticleResampler {
public:
  /**
   * Destructor.
   */
  virtual ~ParticleResampler();

  /**
   * Resample particle set.
   *
   * @param p Particle set to resample.
   *
   * @return Resampled particle set.
   */
  virtual indii::ml::aux::DiracMixturePdf resample(
      indii::ml::aux::DiracMixturePdf& p) = 0;

};

    }
  }
}

#endif
