#ifndef INDII_ML_FILTER_ADDITIVENOISEPARTICLERESAMPLER_HPP
#define INDII_ML_FILTER_ADDITIVENOISEPARTICLERESAMPLER_HPP

#include "ParticleResampler.hpp"
#include "../aux/GaussianPdf.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Particle resampler with independent additive noise source.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 520 $
 * @date $Date: 2008-08-15 14:22:21 +0100 (Fri, 15 Aug 2008) $
 *
 * @param P Type of independent resampling noise source.
 *
 * Produces a new approximation of a weighted sample set by adding
 * noise from an independent source. This would usually be used after
 * resampling with a weight-based resampler such as
 * StratifiedParticleResampler, a la Condensation @ref Isard1998
 * "(Isard & Blake 1998)".
 */
template <class P = indii::ml::aux::GaussianPdf>
class AdditiveNoiseParticleResampler : public ParticleResampler {
public:
  /**
   * Constructor.
   *
   * @param r Independent resampling noise source.
   */
  AdditiveNoiseParticleResampler(const P& r);
  
  /**
   * Destructor.
   */
  virtual ~AdditiveNoiseParticleResampler();

  /**
   * Resample particle set.
   *
   * @param p Particle set to resample.
   *
   * The resampled particle set is constructed by taking each weighted
   * sample \f$(\mathbf{s}^{(i)},\pi^{(i)})\f$ of @p p and adding a
   * sample \f$\mathbf{r}^{(i)}\f$ from the independent reampling
   * noise source, giving
   * \f$(\mathbf{s}^{(i)}+\mathbf{r}^{(i)},\pi^{(i)})\f$.
   *
   * @return Resampled particle set.
   */
  virtual indii::ml::aux::DiracMixturePdf resample(
      indii::ml::aux::DiracMixturePdf& p);

private:
  /**
   * Independent resampling noise source.
   */
  P r;

};

    }
  }
}

template <class P>
indii::ml::filter::AdditiveNoiseParticleResampler<P>::AdditiveNoiseParticleResampler(
    const P& r) : r(r) {
  //
}

template <class P>
indii::ml::filter::AdditiveNoiseParticleResampler<P>::~AdditiveNoiseParticleResampler() {
  //
}

template <class P>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::AdditiveNoiseParticleResampler<P>::resample(
    indii::ml::aux::DiracMixturePdf& p) {
  indii::ml::aux::DiracMixturePdf q(p.getDimensions());

  const unsigned int P_local = p.getNumComponents();
  unsigned int i;

  for (i = 0; i < P_local; i++) {
    q.addComponent(aux::DiracPdf(p.getComponent(i) + r.sample()),
        p.getWeight());
  }

  return q;
}

#endif

