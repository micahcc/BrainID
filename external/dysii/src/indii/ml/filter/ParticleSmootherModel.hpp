#ifndef INDII_ML_FILTER_PARTICLESMOOTHERMODEL_HPP
#define INDII_ML_FILTER_PARTICLESMOOTHERMODEL_HPP

#include "../aux/matrix.hpp"
#include "../aux/DiracMixturePdf.hpp"
#include "ParticleFilterModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * ParticleSmoother compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 437 $
 * @date $Date: 2008-04-28 00:23:52 +0100 (Mon, 28 Apr 2008) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class ParticleSmootherModel : public virtual ParticleFilterModel<T> {
public:
  /**
   * Destructor.
   */
  virtual ~ParticleSmootherModel() = 0;

  /**
   * Calculates the matrix \f$\alpha(t)\f$. \f$\alpha^{(j,i)}(t) =
   * p\big(\mathbf{s}^{(j)}(t + \Delta
   * t)\,|\,\mathbf{s}^{(i)}(t)\big)\f$, where \f$\mathbf{s}^{(j)}(t +
   * \Delta t)\f$ is the \f$j\f$th particle at time \f$t + \Delta t\f$
   * and \f$\mathbf{s}^{(i)}(t)\f$ is the \f$i\f$th particle at time
   * \f$t\f$.
   *
   * @param p_xtn_ytn \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_n)\big)\f$.
   * @param p_xtnp1_ytnp1 \f$P\big(\mathbf{x}(t_{n+1})\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_{n+1})\big)\f$.
   * @param start \f$t_n\f$; start time.
   * @param delta \f$\Delta t = t_{n+1} - t_n\f$; change in time.
   *
   * @return \f$\alpha(t)\f$.
   *
   * Particles for the distributions may be obtained with calls to
   * indii::ml::aux::DiracMixturePdf::getComponents().
   *
   * @note The implementation should not assume that there are \f$P\f$
   * particles in @c p_xtn_ytn and @c p_xtnp1_ytnp1. Instead, use
   * indii::ml::aux::DiracMixturePdf::getNumComponents() to determine
   * this. This is particularly important when working in a parallel
   * environment, where particles are divided up amongst multiple
   * calls to this method.
   */
  virtual indii::ml::aux::sparse_matrix alpha(
      const indii::ml::aux::DiracMixturePdf& p_xtn_ytn,
      const indii::ml::aux::DiracMixturePdf& p_xtnp1_ytnp1,
      const T start, const T delta) = 0;

  /**
   * Perform precalculations for \f$\alpha\f$ matrix.
   *
   * In a parallel environment, each smoothing step may require
   * multiple calls to the alpha() method of the model with the same
   * first argument. To allow for precalculations based on this first
   * argument, the alphaPrecalculate() method is called before each
   * change to the argument.
   *
   * For example, in the case of a model with simple Gaussian additive
   * noise, this method could be used to propagate each of the
   * particles at time \f$t_n\f$ to time \f$t_{n+1}\f$ without noise,
   * obtaining the mean of a Gaussian with covariance equivalent to
   * the system noise. Within the alpha() method, these Gaussians can
   * then be used to quickly calculate densities without having to
   * transition particles again during each call to alpha().
   *
   * There is no requirement for this method to do anything at all, or
   * even to be implemented in derived classes. It exists purely for
   * optimisation, and is not required for correctness.
   *
   * @param p_xtn_ytn As p_xtn_ytn of alpha().
   * @param start As start of alpha().
   * @param delta As delta of alpha().
   */
  virtual void alphaPrecalculate(
      const indii::ml::aux::DiracMixturePdf& p_xtn_ytn, const T start,
      const T delta);

};

    }
  }
}

template <class T>
indii::ml::filter::ParticleSmootherModel<T>::~ParticleSmootherModel() {
  //
}

template <class T>
void indii::ml::filter::ParticleSmootherModel<T>::alphaPrecalculate(
    const indii::ml::aux::DiracMixturePdf& p_xtn_ytn, const T start,
    const T delta) {
  //
}

#endif
