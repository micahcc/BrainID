#ifndef INDII_ML_FILTER_PARTICLESMOOTHER_HPP
#define INDII_ML_FILTER_PARTICLESMOOTHER_HPP

#include "Smoother.hpp"
#include "ParticleResampler.hpp"
#include "ParticleSmootherModel.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Forward-backward particle smoother.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 560 $
 * @date $Date: 2008-09-08 23:40:30 +0100 (Mon, 08 Sep 2008) $
 *
 * ParticleSmoother is suitable for models with nonlinear transition
 * and measurement functions, approximating state and noise with
 * indii::ml::aux::DiracMixturePdf distributions. It is particularly
 * suitable in situations where an appropriate backwards transition
 * function cannot be defined, such as for transition functions
 * defined as convergent differential equations that become divergent
 * when reversed.
 * 
 * @see indii::ml::filter for general usage guidelines.
 *
 * @section ParticleSmoother_details Details
 *
 * The implementation here is of the second algorithm described in
 * @ref Isard1998 "Isard & Blake (1998)". This reweights particles
 * obtained during the forwards pass rather than performing a
 * backwards filtering pass to obtain new particles. While this means
 * a backwards transition function need not be defined for the model,
 * the drawback is that the approach is computationally and spatially
 * expensive -- \f$O(P^2)\f$ in the number of particles \f$P\f$ in
 * both cases.
 *
 * The forwards pass provides a weighted sample set
 * \f$\{(\mathbf{s}_t^{(i)},\pi_t^{(i)})\}\f$ at each time point \f$t =
 * t_1,\ldots,t_T\f$ for \f$i = 1,\ldots,P\f$. Initialising with \f$\psi_{t_T}
 * = \pi_{t_T}\f$, the backwards step to calculate weights at time \f$t_n\f$
 * is as follows:
 *
 * \f{eqnarray*}
 * \alpha_{t_n}^{(i,j)} &=& p(\mathbf{x}({t_{n+1}}) =
 * \mathbf{s}_{t_{n+1}}^{(i)}\,|\,\mathbf{x}({t_n}) =
 * \mathbf{s}_{t_n}^{(j)}) \\
 * \mathbf{\gamma}_{t_n} &=& \alpha_{t_n} \mathbf{\pi}_{t_n} \\
 * \mathbf{\delta}_{t_n} &=& \alpha_{t_n}^T(\mathbf{\psi}_{t_{n+1}}
 * \oslash \mathbf{\gamma}_{t_n})\\
 * \mathbf{\psi}_{t_n} &=& \mathbf{\pi}_{t_n} \otimes
 * \mathbf{\delta}_{t_n}
 * \f}
 *
 * where \f$\oslash\f$ is element-wise division and \f$\otimes\f$
 * element-wise multiplication.
 *
 * These are then normalised so that \f$\sum \psi_{t_n}^{(i)} = 1\f$
 * and the smoothed result
 * \f$\{(\mathbf{s}_{t_n}^{(i)},\psi_{t_n}^{(i)})\}\f$ for \f$i =
 * 1,\ldots,P\f$ is stored.
 *
 * This implementation performs calculations in the above matrix form
 * to take advantage of low-level optimisations in the matrix
 * library. It also uses a sparse matrix implementation of
 * \f$\alpha\f$ to alleviate spatial complexity somewhat.
 *
 * @section ParticleSmoother_references References
 *
 * @anchor Isard1998
 * Isard, M. & Blake, A. A smoothing %filter for
 * Condensation. <i>Proceedings of the 5th European Conference on
 * Computer Vision</i>, <b>1998</b>, 1, 767-781.
 */
template <class T = unsigned int>
class ParticleSmoother : public Smoother<T,indii::ml::aux::DiracMixturePdf> {
public:
  /**
   * Create new particle smoother.
   * 
   * @param model Model to estimate.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  ParticleSmoother(ParticleSmootherModel<T>* model,
      const T tT, const indii::ml::aux::DiracMixturePdf& p_xT);

  /**
   * Destructor.
   */
  virtual ~ParticleSmoother();

  /**
   * Get the model being estimated.
   *
   * @return The model being estimated.
   */
  virtual ParticleSmootherModel<T>* getModel();

  /**
   * Rewind system to time of previous measurement and
   * smooth.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the
   * system. This must be less than the current time \f$t_{n+1}\f$.
   * @param p_xtn_ytn \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n})\f$; filter
   * density at time \f$t_n\f$.
   */
  virtual void smooth(const T tn,
      const indii::ml::aux::DiracMixturePdf& p_xtn_ytn);

  virtual indii::ml::aux::DiracMixturePdf smoothedMeasure();

  /**
   * Resample the smoothed state.
   *
   * @see ParticleFilter::resample()
   */
  void smoothedResample(ParticleResampler* resampler);

private:
  /**
   * Model.
   */
  ParticleSmootherModel<T>* model;

};

    }
  }
}

#include "StratifiedParticleResampler.hpp"

#include <assert.h>
#include <vector>

#include "boost/numeric/ublas/operation.hpp"
#include "boost/numeric/ublas/operation_sparse.hpp"

template <class T>
indii::ml::filter::ParticleSmoother<T>::ParticleSmoother(
    ParticleSmootherModel<T>* model, const T tT,
    const indii::ml::aux::DiracMixturePdf& p_xT) :
    Smoother<T,indii::ml::aux::DiracMixturePdf>(tT, p_xT), model(model) {
  //
}

template <class T>
indii::ml::filter::ParticleSmoother<T>::~ParticleSmoother() {
  //
}

template <class T>
indii::ml::filter::ParticleSmootherModel<T>*
    indii::ml::filter::ParticleSmoother<T>::getModel() {
  return model;
}

template <class T>
void indii::ml::filter::ParticleSmoother<T>::smooth(const T tn,
    const indii::ml::aux::DiracMixturePdf& p_xtn_ytn) {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  /* pre-condition */
  assert (tn < this->tn);

  const T tnp1 = this->tn;
  aux::DiracMixturePdf& p_xtnp1_ytT = this->p_xtn_ytT;
  
  const unsigned int N = p_xtn_ytn.getDimensions();
  const unsigned int P1 = p_xtn_ytn.getSize();
  const unsigned int P2 = p_xtnp1_ytT.getSize();
  const T del = tnp1 - tn;
  unsigned int i;

  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();
  unsigned int k;

  const aux::vector& pi = p_xtn_ytn.getWeights();
  aux::vector psi(p_xtnp1_ytT.getWeights());
  aux::vector gamma(P2);
  aux::vector delta(P1);
  std::vector<aux::sparse_matrix> alphas;

  /* calculate \f$\alpha\f$ */
  model->alphaPrecalculate(p_xtn_ytn, tn, del);
  for (k = 0; k < size; k++) {
    alphas.push_back(model->alpha(p_xtn_ytn, p_xtnp1_ytT, tn, del));
    indii::ml::aux::rotate(p_xtnp1_ytT);
  }

  /* calculate \f$\gamma\f$ */
  gamma.clear();
  for (k = 0; k < size; k++) {
    ublas::axpy_prod(alphas[k], pi, gamma, false);
    indii::ml::aux::rotate(gamma);
  }

  /* calculate \f$\delta\f$ */
  delta.clear();
  psi = element_div(psi, gamma);
  for (k = 0; k < size; k++) {
    ublas::axpy_prod(psi, alphas[k], delta, false);
    indii::ml::aux::rotate(psi);
  }

  /* calculate \f$\psi_{t_{n+1}}\f$ */
  psi.resize(P1, false);
  noalias(psi) = element_prod(pi, delta);

  /* build smoothed distribution */
  this->p_xtn_ytT = p_xtn_ytn;
  this->p_xtn_ytT.setWeights(psi);

  /* update state */
  this->tn = tn;
}

template <class T>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::ParticleSmoother<T>::smoothedMeasure() {
  namespace aux = indii::ml::aux;

  unsigned int i;
  StratifiedParticleResampler resampler;
  aux::DiracMixturePdf resampled(resampler.resample(
      Smoother<T,aux::DiracMixturePdf>::getSmoothedState()));
  aux::DiracMixturePdf p_ytn_xtT(model->getMeasurementSize());

  for (i = 0; i < resampled.getSize(); i++) {
    p_ytn_xtT.add(model->measure(resampled.get(i)));
  }  
  
  return p_ytn_xtT;
}

template <class T>
void indii::ml::filter::ParticleSmoother<T>::smoothedResample(
    ParticleResampler* resampler) {
  indii::ml::aux::DiracMixturePdf resampled(resampler->resample(
      Smoother<T,aux::DiracMixturePdf>::getSmoothedState()));
  Smoother<T,aux::DiracMixturePdf>::setSmoothedState(resampled);
}

#endif

