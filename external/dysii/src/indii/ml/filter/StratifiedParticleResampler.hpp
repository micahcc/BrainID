#ifndef INDII_ML_FILTER_STRATIFIEDPARTICLERESAMPLER_HPP
#define INDII_ML_FILTER_STRATIFIEDPARTICLERESAMPLER_HPP

#include "ParticleResampler.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Stratified particle resampler.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 518 $
 * @date $Date: 2008-08-15 14:01:35 +0100 (Fri, 15 Aug 2008) $
 *
 * Produces a new approximation of a weighted sample set using a set
 * of equally weighted samples, possibly with duplication. Based on
 * the deterministic stratified resampling scheme of @ref
 * Kitagawa1996 "(Kitagawa 1996)", without sorting.
 *
 * @section StratifiedParticleResampler_references References
 *
 * @anchor Kitagawa1996
 * Kitagawa, G. Monte Carlo %Filter and %Smoother for Non-Gaussian
 * Nonlinear State Space Models. <i>Journal of Computational and
 * Graphical Statistics</i>, <b>1996</b>, 5, 1-25.
 */
class StratifiedParticleResampler : public ParticleResampler {
public:
  /**
   * Constructor.
   *
   * @param P Number of particles to resample from each distribution
   * supplied to resample(). If zero, will resample the same number of
   * particles as the distribution supplied.
   */
  StratifiedParticleResampler(const unsigned int P = 0);

  /**
   * Destructor.
   */
  virtual ~StratifiedParticleResampler();

  /**
   * Set the number of particles to resample.
   *
   * @param P Number of particles to resample from each distribution
   * supplied to resample(). If zero, will resample the same number of
   * particles as the distribution supplied.
   */
  void setNumParticles(const unsigned int P = 0);

  /**
   * Resample the distribution. This produces a new approximation of
   * the same distribution using a set of equally weighted sample
   * points. Sample points are selected using the deterministic
   * resampling method given in the appendix of @ref Kitagawa1996
   * "Kitagawa (1996)".
   *
   * @return The resampled distribution.
   */
  virtual indii::ml::aux::DiracMixturePdf resample(
      indii::ml::aux::DiracMixturePdf& p);

private:
  /**
   * Number of particles to resample from each distribution.
   */
  unsigned int P;

};

    }
  }
}

#endif

