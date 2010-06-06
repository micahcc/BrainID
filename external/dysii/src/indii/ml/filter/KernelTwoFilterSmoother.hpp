#ifndef INDII_ML_FILTER_KERNELTWOFILTERSMOOTHER_HPP
#define INDII_ML_FILTER_KERNELTWOFILTERSMOOTHER_HPP

#include "Smoother.hpp"
#include "ParticleResampler.hpp"
#include "KernelTwoFilterSmootherModel.hpp"
#include "flags.hpp"
#include "../aux/Almost2Norm.hpp"
#include "../aux/AlmostGaussianKernel.hpp"
#include "../aux/MedianPartitioner.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Kernel two-filter smoother.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 589 $
 * @date $Date: 2008-12-15 17:46:51 +0000 (Mon, 15 Dec 2008) $
 *
 * @param T The type of time.
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param ST Partitioner type.
 *
 * @see KernelForwardBackwardSmoother for further information, and 
 * indii::ml::filter for general usage guidelines.
 *
 * @section KernelTwoFilterSmoother_references References
 *
 * @anchor Murray2009
 * Murray, L.M. (2009) Bayesian Learning of Continuous Time Dynamical Systems
 * (with applications in Functional Magnetic Resonance Imaging). PhD thesis.
 * Online at http://www.indii.org/research/.
 */
template <class T = unsigned int,
    class NT = indii::ml::aux::Almost2Norm,
    class KT = indii::ml::aux::AlmostGaussianKernel,
    class ST = indii::ml::aux::MedianPartitioner>
class KernelTwoFilterSmoother :
    public Smoother<T,indii::ml::aux::DiracMixturePdf> {
public:
  /**
   * Constructor, with measurement at starting time.
   * 
   * @param model Model to estimate.
   * @param N The kernel density norm.
   * @param K The kernel density kernel.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   * @param ytT \f$\mathbf{y}_T\f$; measurement at time \f$t_T\f$.
   * @param flags Optimisation flags for calculation of the initial
   * backward likelihood. Only NO_STANDARDISATION is relevant here.
   */
  KernelTwoFilterSmoother(KernelTwoFilterSmootherModel<T>* model,
      const NT& N, const KT& K, const T tT,
      const indii::ml::aux::DiracMixturePdf& p_xT,
      const indii::ml::aux::vector& ytT,
      const unsigned int flags = 0);

  /**
   * Constructor, without measurement at starting time.
   * 
   * @param model Model to estimate.
   * @param N The kernel density norm.
   * @param K The kernel density kernel.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   * @param flags Optimisation flags for calculation of the initial
   * backward likelihood. Only NO_STANDARDISATION is relevant here.
   */
  KernelTwoFilterSmoother(KernelTwoFilterSmootherModel<T>* model,
      const NT& N, const KT& K, const T tT,
      const indii::ml::aux::DiracMixturePdf& p_xT,
      const unsigned int flags = 0);

  /**
   * Destructor.
   */
  virtual ~KernelTwoFilterSmoother();

  /**
   * Get the model.
   *
   * @return The model.
   */
  virtual KernelTwoFilterSmootherModel<T>* getModel();

  /**
   * Step back in time and smooth, with measurement.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the system. This must
   * be less than the current time \f$t_{n+1}\f$.
   * @param ytn \f$\mathbf{y}_n\f$; the measurement at time \f$t_n\f$.
   * @param p_xtn_ytnm1 \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n-1})\f$;
   * the uncorrected %filter density at time \f$t_n\f$.
   * @param q_xtn \f$q(\mathbf{x}_n)\f$; proposal distribution for
   * importance sampling of the smooth density at time \f$t_n\f$.
   * @param flags Optimisation flags.
   */
  void smooth(const T tn,
      const indii::ml::aux::vector& ytn,
      indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
      indii::ml::aux::Pdf& q_xtn,
      const unsigned int flags = 0);

  /**
   * Step back in time and smooth, without measurement.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the system. This must
   * be less than the current time \f$t_{n+1}\f$.
   * @param p_xtn_ytnm1 \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n-1})\f$;
   * the uncorrected %filter density at time \f$t_n\f$.
   * @param q_xtn \f$q(\mathbf{x}_n)\f$; proposal distribution for
   * importance sampling of the smooth density at time \f$t_n\f$.
   * @param flags Optimisation flags.
   */
  void smooth(const T tn,
      indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
      indii::ml::aux::Pdf& q_xtn,
      const unsigned int flags = 0);

  /**
   * Step back in time and smooth, with measurement, and uncorrected
   * filter density as proposal distribution.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the system. This must
   * be less than the current time \f$t_{n+1}\f$.
   * @param ytn \f$\mathbf{y}_n\f$; the measurement at time \f$t_n\f$.
   * @param p_xtn_ytnm1 \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n-1})\f$;
   * the uncorrected %filter density at time \f$t_n\f$.
   * @param flags Optimisation flags.
   */
  void smooth(const T tn,
      const indii::ml::aux::vector& ytn,
      indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
      const unsigned int flags = 0);

  /**
   * Step back in time and smooth, without measurement, and uncorrected
   * filter density as proposal distribution.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the system. This must
   * be less than the current time \f$t_{n+1}\f$.
   * @param p_xtn_ytnm1 \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n-1})\f$;
   * the uncorrected %filter density at time \f$t_n\f$.
   * @param flags Optimisation flags.
   */
  void smooth(const T tn,
      indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
      const unsigned int flags = 0);

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
  KernelTwoFilterSmootherModel<T>* model;

  /**
   * \f$\|\mathbf{x}\|_p\f$; the norm.
   */
  NT N;
  
  /**
   * \f$K(\|\mathbf{x}\|_p) \f$; the density kernel.
   */
  KT K;

  /**
   * Number of samples to use.
   */
  unsigned int P;

  /**
   * Proposal distribution samples.
   */
  indii::ml::aux::DiracMixturePdf q_xtns;
  
  /**
   * Proposal distribution sample densities.
   */
  indii::ml::aux::vector q_ptns;
  
  /**
   * Proposal distribution sample propagations.
   */
  indii::ml::aux::DiracMixturePdf q_xtnp1s;
  
  /**
   * Time difference for last proposal sample propagations.
   */
  T q_delta;

  /**
   * \f$p(\mathbf{y}_{n:T}\,|\,\mathbf{x}_n)\f$; backward likelihood.
   */
  DiracMixturePdf p_ytn_xtn;
  
};

    }
  }
}

#include "StratifiedParticleResampler.hpp"
#include "../aux/kde.hpp"

#include <assert.h>

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::KernelTwoFilterSmoother(
    KernelTwoFilterSmootherModel<T>* model, const NT& N, const KT& K,
    const T tT, const indii::ml::aux::DiracMixturePdf& p_xT,
    const indii::ml::aux::vector& ytT, const unsigned int flags) :
    Smoother<T,indii::ml::aux::DiracMixturePdf>(tT, p_xT),
    model(model),
    N(N),
    K(K),
    q_xtns(model->getStateSize()),
    q_xtnp1s(model->getStateSize()),
    p_ytn_xtn(p_xT) {
  namespace aux = indii::ml::aux;

  unsigned int i;
    
  P = p_xT.getDistributedSize();

  /* initialise backward likelihood */
  aux::vector ws(p_ytn_xtn.getSize());
  if (flags & NO_STANDARDISATION) {
    aux::KDTree<ST> tree(&p_ytn_xtn);
    noalias(ws) = aux::distributedSelfTreeDensity(tree,
        p_ytn_xtn.getWeights(), N, K);
  } else {
    aux::DiracMixturePdf q(p_ytn_xtn);
    q.distributedStandardise();

    aux::KDTree<ST> tree(&q);
    noalias(ws) = aux::distributedSelfTreeDensity(tree,
        q.getWeights(), N, K);
  }
  for (i = 0; i < p_ytn_xtn.getSize(); i++) {
    ws(i) /= model->weight(p_ytn_xtn.get(i), ytT);
  }
      
  p_ytn_xtn.setWeights(element_div(p_ytn_xtn.getWeights(), ws));
}

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::KernelTwoFilterSmoother(
    KernelTwoFilterSmootherModel<T>* model, const NT& N, const KT& K,
    const T tT, const indii::ml::aux::DiracMixturePdf& p_xT,
    const unsigned int flags) :
    Smoother<T,indii::ml::aux::DiracMixturePdf>(tT, p_xT),
    model(model),
    N(N),
    K(K),
    q_xtns(model->getStateSize()),
    q_xtnp1s(model->getStateSize()),
    p_ytn_xtn(p_xT) {
  namespace aux = indii::ml::aux;

  unsigned int i;
    
  P = p_xT.getDistributedSize();

  /* initialise backward likelihood */
  aux::vector ws(p_ytn_xtn.getSize());
  if (flags & NO_STANDARDISATION) {
    aux::KDTree<ST> tree(&p_ytn_xtn);
    noalias(ws) = aux::distributedSelfTreeDensity(tree,
        p_ytn_xtn.getWeights(), N, K);
  } else {
    aux::DiracMixturePdf q(p_ytn_xtn);
    q.distributedStandardise();

    aux::KDTree<ST> tree(&q);
    noalias(ws) = aux::distributedSelfTreeDensity(tree,
        q.getWeights(), N, K);
  }
      
  p_ytn_xtn.setWeights(element_div(p_ytn_xtn.getWeights(), ws));
}

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::~KernelTwoFilterSmoother() {
  //
}

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelTwoFilterSmootherModel<T>*
    indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::getModel() {
  return model;
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::smooth(
    const T tn,
    const indii::ml::aux::vector& ytn,
    indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
    indii::ml::aux::Pdf& q_xtn,
    const unsigned int flags) {
  /* pre-condition */
  assert (q_xtn.getDimensions() == p_xtn_ytnm1.getDimensions());
  
  const unsigned int D = model->getStateSize();
  vector x(D);
  const T del = this->tn - tn;
  DiracMixturePdf& p_ytnp1_xtnp1 = this->p_ytn_xtn;
  unsigned int P_local = aux::shareOf(P);
  unsigned int i;

  /* pick apart flags */
  bool sameProposal = flags & SAME_PROPOSAL;
  bool samePropagations = flags & SAME_PROPAGATIONS;
  bool noResampling = flags & NO_RESAMPLING;
  bool noStandardisation = flags & NO_STANDARDISATION;

  /* sample particles from proposal */
  if (!sameProposal || q_xtns.getSize() == 0) {
    q_xtns.clear();
    q_ptns.resize(P_local);
    for (i = 0; i < P_local; i++) {
      noalias(x) = q_xtn.sample();
      q_xtns.add(x);
      q_ptns(i) = q_xtn.densityAt(x);
    }
  }

  /* propagate sample particles */
  if (!samePropagations || !sameProposal || this->q_delta != del ||
      q_xtns.getSize() != q_xtnp1s.getSize()) {
    this->q_delta = del;
    q_xtnp1s.clear();
    for (i = 0; i < q_xtns.getSize(); i++) {
      noalias(x) = model->transition(q_xtns.get(i), tn, del);
      q_xtnp1s.add(x, q_xtns.getWeight(i));
    }
  }

  /* likelihood evaluation */
  aux::vector l(P_local);
  if (ytn.size() > 0) {
    for (i = 0; i < P_local; i++) {
      l(i) = model->weight(q_xtns.get(i), ytn);
    }
  } else {
    l = aux::scalar_vector(P_local, 1.0);
  }

  aux::vector a(P_local), b(P_local), beta(P_local), psi(P_local);
  if (noStandardisation) {
    /* uncorrected filter density evaluation */
    {
      p_xtn_ytnm1.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q_xtns);
      aux::KDTree<ST> targetTree(&p_xtn_ytnm1);
  
      noalias(a) = aux::distributedDualTreeDensity(queryTree, targetTree,
          p_xtn_ytnm1.getWeights(), N, K);
    }
    
    /* likelihood evaluation */
    {
      p_ytnp1_xtnp1.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q_xtnp1s);
      aux::KDTree<ST> targetTree(&p_ytnp1_xtnp1);

      noalias(b) = aux::distributedDualTreeDensity(queryTree, targetTree,
          p_ytnp1_xtnp1.getWeights(), N, K);
    }
  } else {
    /* uncorrected filter density evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      
      noalias(mu) = p_xtn_ytnm1.getDistributedExpectation();
      noalias(L) = p_xtn_ytnm1.getDistributedStandardDeviation();

      DiracMixturePdf q(q_xtns);
      q.standardise(mu, L);
      p_xtn_ytnm1.standardise(mu, L);

      p_xtn_ytnm1.redistributeBySpace();    
      aux::KDTree<ST> queryTree(&q);
      aux::KDTree<ST> targetTree(&p_xtn_ytnm1);
  
      noalias(a) = aux::distributedDualTreeDensity(queryTree, targetTree,
          p_xtn_ytnm1.getWeights(), N, K);
    }
    
    /* likelihood evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      
      noalias(mu) = p_ytnp1_xtnp1.getDistributedExpectation();
      noalias(L) = p_ytnp1_xtnp1.getDistributedStandardDeviation();

      DiracMixturePdf q(q_xtnp1s);
      q.standardise(mu, L);
      p_ytnp1_xtnp1.standardise(mu, L);

      p_ytnp1_xtnp1.redistributeBySpace();    
      aux::KDTree<ST> queryTree(&q);
      aux::KDTree<ST> targetTree(&p_ytnp1_xtnp1);
  
      noalias(b) = aux::distributedDualTreeDensity(queryTree, targetTree,
          p_ytnp1_xtnp1.getWeights(), N, K);
    }  
  }
  
  noalias(beta) = element_div(element_prod(l,b), q_ptns);
  beta = element_prod(beta, q_xtns.getWeights());
  noalias(psi) = element_prod(beta, a);

  /* rebuild densities/likelihoods */
  this->p_ytn_xtn = q_xtns;
  this->p_ytn_xtn.setWeights(beta);

  this->p_xtn_ytT = q_xtns;
  this->p_xtn_ytT.setWeights(psi);

  /* update state */
  this->tn = tn;
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::smooth(
    const T tn,
    indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
    indii::ml::aux::Pdf& q_xtn,
    const unsigned int flags) {
  aux::vector ytn;
  smooth(tn, ytn, p_xtn_ytnm1, q_xtn, flags);
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::smooth(
    const T tn,
    const indii::ml::aux::vector& ytn,
    indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
    const unsigned int flags) {
  const unsigned int D = model->getStateSize();
  vector x(D);
  const T del = this->tn - tn;
  DiracMixturePdf& p_ytnp1_xtnp1 = this->p_ytn_xtn;
  unsigned int P_local = p_xtn_ytnm1.getSize();
  unsigned int i;

  /* pick apart flags */
  bool noResampling = flags & NO_RESAMPLING;
  bool noStandardisation = flags & NO_STANDARDISATION;

  /* propagate sample particles */
  if (!noResampling) {
    this->q_delta = del;
    q_xtnp1s.clear();
    for (i = 0; i < p_xtn_ytnm1.getSize(); i++) {
      noalias(x) = model->transition(p_xtn_ytnm1.get(i), tn, del);
      q_xtnp1s.add(x, p_xtn_ytnm1.getWeight(i));
    }
  } else {
    assert (p_xtn_ytnm1.getSize() == p_ytnp1_xtnp1.getSize());
  }

  /* likelihood evaluation */
  aux::vector l(P_local);
  if (ytn.size() > 0) {
    for (i = 0; i < P_local; i++) {
      l(i) = model->weight(p_xtn_ytnm1.get(i), ytn);
    }
  } else {
    l = aux::scalar_vector(P_local, 1.0);
  }

  aux::vector a(P_local), b(P_local), beta(P_local), psi(P_local);
  const aux::vector& pi = p_xtn_ytnm1.getWeights();
  if (noStandardisation) {
    /* uncorrected filter density evaluation */
    {
      aux::KDTree<ST> tree(&p_xtn_ytnm1);
  
      noalias(a) = aux::distributedSelfTreeDensity(tree,
          p_xtn_ytnm1.getWeights(), N, K);
    }
    
    /* likelihood evaluation */
    {
      if (noResampling) {
        aux::KDTree<ST> tree(&p_ytnp1_xtnp1);
        
        noalias(b) = aux::distributedSelfTreeDensity(tree,
            p_ytnp1_xtnp1.getWeights(), N, K);
      } else {
        p_ytnp1_xtnp1.redistributeBySpace();
        aux::KDTree<ST> queryTree(&q_xtnp1s);
        aux::KDTree<ST> targetTree(&p_ytnp1_xtnp1);

        noalias(b) = aux::distributedDualTreeDensity(queryTree, targetTree,
            p_ytnp1_xtnp1.getWeights(), N, K);
      }
    }
  } else {
    /* uncorrected filter density evaluation */
    {
      DiracMixturePdf q(p_xtn_ytnm1);
      q.distributedStandardise();
      aux::KDTree<ST> tree(&q);

      noalias(a) = aux::distributedSelfTreeDensity(tree, q.getWeights(),
          N, K);
    }
    
    /* likelihood evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      
      noalias(mu) = p_ytnp1_xtnp1.getDistributedExpectation();
      noalias(L) = p_ytnp1_xtnp1.getDistributedStandardDeviation();
      p_ytnp1_xtnp1.standardise(mu, L);

      if (noResampling) {
        aux::KDTree<ST> tree(&p_ytnp1_xtnp1);
        
        noalias(b) = aux::distributedSelfTreeDensity(tree,
            p_ytnp1_xtnp1.getWeights(), N, K);
      } else {
        DiracMixturePdf q(q_xtnp1s);
        q.standardise(mu, L);

        p_ytnp1_xtnp1.redistributeBySpace();
        aux::KDTree<ST> queryTree(&q);
        aux::KDTree<ST> targetTree(&p_ytnp1_xtnp1);
  
        noalias(b) = aux::distributedDualTreeDensity(queryTree, targetTree,
            p_ytnp1_xtnp1.getWeights(), N, K);
      }
    }
  }
  
  noalias(psi) = element_prod(element_prod(l, b), pi);
  noalias(beta) = element_div(psi, a);

  /* rebuild densities/likelihoods */
  this->p_ytn_xtn = p_xtn_ytnm1;
  this->p_ytn_xtn.setWeights(beta);

  this->p_xtn_ytT = p_xtn_ytnm1;
  this->p_xtn_ytT.setWeights(psi);
  
  /* update state */
  this->tn = tn;
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::smooth(
    const T tn,
    indii::ml::aux::DiracMixturePdf& p_xtn_ytnm1,
    const unsigned int flags) {
  aux::vector ytn;
  smooth(tn, ytn, p_xtn_ytnm1, flags);
}

template <class T, class NT, class KT, class ST>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::smoothedMeasure() {
  namespace aux = indii::ml::aux;
    
  unsigned int i;
  StratifiedParticleResampler resampler;
  aux::DiracMixturePdf resampled(resampler.resample(
      this->getSmoothedState()));
  indii::ml::aux::DiracMixturePdf p_ytn_xtT(model->getMeasurementSize());
  
  for (i = 0; i < resampled.getSize(); i++) {
    p_ytn_xtT.add(model->measure(resampled.get(i)));
  }

  return p_ytn_xtT;
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelTwoFilterSmoother<T,NT,KT,ST>::smoothedResample(
    ParticleResampler* resampler) {
  indii::ml::aux::DiracMixturePdf resampled(resampler->resample(
      this->getSmoothedState()));
  this->setSmoothedState(resampled);
}

#endif

