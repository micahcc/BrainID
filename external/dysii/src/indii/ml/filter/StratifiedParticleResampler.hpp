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
   * Set to use the original deterministic resample method 
   */
  void useDeterministic() { method = DETERMINISTIC; };
  
  /**
   * Set to use DiracMixture's function for drawing particle
   */
  void useDiracMixture() { method = MIXTURE;} ; 

  /**
   * Use the the custom random generation method
   */
  void useCustom1() { method = CUSTOM1; };
  void useCustom2() { method = CUSTOM2; };


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
  
  /**
   * Method of Resampling
   * Deterministic is a heuristic method of deterministically resampling
   * mixture uses MixturePdf's ability to sample from the distribution
   * custom is my own version of sampling from the distribution, 
   *            using binary search on the cumulative weights O(NlogN)
   */
  unsigned int method;
  enum {DETERMINISTIC, MIXTURE, CUSTOM1, CUSTOM2};
  
  /**
   * persistent random number generation, to prevent re-seeding
   */
  gsl_rng* rng;

  void resample_custom(const indii::ml::aux::DiracMixturePdf& p,
              indii::ml::aux::DiracMixturePdf& resampled);
  void resample_mixture(const indii::ml::aux::DiracMixturePdf& p,
              indii::ml::aux::DiracMixturePdf& resampled);
  void resample_deterministic(const indii::ml::aux::DiracMixturePdf& p,
              indii::ml::aux::DiracMixturePdf& resampled);

};

    }
  }
}

#endif

