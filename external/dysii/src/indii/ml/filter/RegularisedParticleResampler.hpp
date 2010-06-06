#ifndef INDII_ML_FILTER_REGULARISEDPARTICLERESAMPLER_HPP
#define INDII_ML_FILTER_REGULARISEDPARTICLERESAMPLER_HPP

#include "ParticleResampler.hpp"
#include "../aux/Almost2Norm.hpp"
#include "../aux/AlmostGaussianKernel.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Regularised particle resampler.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 576 $
 * @date $Date: 2008-11-16 15:24:50 +0000 (Sun, 16 Nov 2008) $
 *
 * Adds standardised kernel noise to each particle. Another resampler, such
 * as DeterministicParticleResampler, should usually be applied first.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 */
template <class NT = indii::ml::aux::Almost2Norm,
    class KT = indii::ml::aux::AlmostGaussianKernel>
class RegularisedParticleResampler : public ParticleResampler {
public:
  /**
   * Constructor.
   *
   * @param N The kernel density norm.
   * @param K The kernel density kernel.
   */
  RegularisedParticleResampler(const NT& N, const KT& K);

  /**
   * Destructor.
   */
  virtual ~RegularisedParticleResampler();

  /**
   * Set the kernel density norm.
   *
   * @param N The kernel density norm.
   */
  void setNorm(const NT& N);

  /**
   * Set the kernel density kernel.
   *
   * @param K The kernel density kernel.
   */
  void setKernel(const KT& K);

  /**
   * Resample the distribution.
   *
   * @return The resampled distribution.
   */
  virtual indii::ml::aux::DiracMixturePdf resample(
      indii::ml::aux::DiracMixturePdf& p);

private:
  /**
   * \f$\|\mathbf{x}\|_p\f$; the norm.
   */
  NT N;
  
  /**
   * \f$K(\|\mathbf{x}\|_p) \f$; the density kernel.
   */
  KT K;

};

    }
  }
}

#include "../aux/KernelDensityMixturePdf.hpp"
#include "../aux/KDTree.hpp"

template <class NT, class KT>
indii::ml::filter::RegularisedParticleResampler<NT,KT>::RegularisedParticleResampler(
    const NT& N, const KT& K) : N(N), K(K) {
  //
}

template <class NT, class KT>
indii::ml::filter::RegularisedParticleResampler<NT,KT>::~RegularisedParticleResampler() {
  //
}

template <class NT, class KT>
void indii::ml::filter::RegularisedParticleResampler<NT,KT>::setNorm(const NT& N) {
  this->N = N;
}

template <class NT, class KT>
void indii::ml::filter::RegularisedParticleResampler<NT,KT>::setKernel(const KT& K) {
  this->K = K;
}

template <class NT, class KT>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::RegularisedParticleResampler<NT,KT>::resample(
    indii::ml::aux::DiracMixturePdf& p) {
  namespace aux = indii::ml::aux;

  unsigned int i;
  aux::DiracMixturePdf r(p.getDimensions());
  aux::vector x(p.getDimensions());

  /* standardise particles */
  aux::lower_triangular_matrix sd(p.getDistributedStandardDeviation());

  /* rebuild distribution with kernel noise */
  for (i = 0; i < p.getSize(); i++) {
    noalias(x) = p.get(i) + prod(sd, K.sample() * N.sample(
        p.getDimensions()));
    r.add(x, p.getWeight(i));
  }

  return r;
}

#endif


