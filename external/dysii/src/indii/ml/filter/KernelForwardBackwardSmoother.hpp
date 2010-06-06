#ifndef INDII_ML_FILTER_KERNELFORWARDBACKWARDSMOOTHER_HPP
#define INDII_ML_FILTER_KERNELFORWARDBACKWARDSMOOTHER_HPP

#include "Smoother.hpp"
#include "ParticleResampler.hpp"
#include "KernelForwardBackwardSmootherModel.hpp"
#include "flags.hpp"
#include "../aux/Almost2Norm.hpp"
#include "../aux/AlmostGaussianKernel.hpp"
#include "../aux/MedianPartitioner.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Kernel forward-backward smoother.
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
 * KernelForwardBackwardSmoother is suitable for continuous time systems with
 * nonlinear transition and measurement functions, approximating state and
 * noise with indii::ml::aux::DiracMixturePdf distributions. It is
 * particularly suitable in situations where the transition density is
 * intractable, such as for transition functions defined using Stochastic
 * Differential Equations (SDEs).
 *
 * For ease of use, KernelForwardBackwardSmoother handles proposal
 * distribution sampling and sample propagations internally.
 * 
 * A number of significant optimisations may be triggered using Flags. The
 * use of flags is entirely optional, and considered an advanced feature.
 * Not using flags will trigger the most generally applicable algorithms,
 * suitable in all situations. Using the right flags in the right situation
 * will give significant performance improvements. Using flags in the wrong
 * situation will give erronous results. Be sure to understand the 
 * assumptions implied by a flag, and be certain that those assumptions are
 * suitable, before putting it to use.
 * 
 * @see indii::ml::filter for general usage guidelines.
 *
 * @section KernelForwardBackwardSmoother_references References
 *
 * @anchor Murray2009
 * Murray, L.M. (2009) Bayesian Learning of Continuous Time Dynamical Systems
 * (with applications in Functional Magnetic Resonance Imaging). PhD thesis.
 * Online at http://www.indii.org/research/.
 */
template <class T = unsigned int,
    class NT = Almost2Norm,
    class KT = AlmostGaussianKernel,
    class ST = MedianPartitioner>
class KernelForwardBackwardSmoother :
    public Smoother<T,indii::ml::aux::DiracMixturePdf> {
public:  
  /**
   * Constructor.
   * 
   * @param model Model to estimate.
   * @param N The kernel density norm.
   * @param K The kernel density kernel.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  KernelForwardBackwardSmoother(KernelForwardBackwardSmootherModel<T>* model,
      const NT& N, const KT& K, const T tT,
      const indii::ml::aux::DiracMixturePdf& p_xT);

  /**
   * Destructor.
   */
  virtual ~KernelForwardBackwardSmoother();

  /**
   * Get the model being estimated.
   *
   * @return The model being estimated.
   */
  virtual KernelForwardBackwardSmootherModel<T>* getModel();

  /**
   * Step back in time and smooth.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the
   * system. This must be less than the current time \f$t_{n+1}\f$.
   * @param p_xtn_ytn \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n})\f$;
   * filter density at time \f$t_n\f$. May be modified.
   * @param p_xtnp1_ytn \f$p(\mathbf{x}_{n+1}\,|\,\mathbf{y}_{1:n})\f$;
   * uncorrected filter density at time \f$t_{n+1}\f$. May be modified.
   * @param q_xtn \f$q(\mathbf{x}_n)\f$; proposal distribution.
   * @param flags Optimisation flags.
   *
   * @see Flags for optimisation flags.
   */
  void smooth(const T tn,
      indii::ml::aux::DiracMixturePdf& p_xtn_ytn,
      indii::ml::aux::DiracMixturePdf& p_xtnp1_ytn,
      indii::ml::aux::Pdf& q_xtn,
      const unsigned int flags = 0);

  /**
   * Step back in time and smooth, using %filter density as proposal
   * distribution.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the
   * system. This must be less than the current time \f$t_{n+1}\f$.
   * @param p_xtn_ytn \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n})\f$;
   * filter density at time \f$t_n\f$. May be modified.
   * @param p_xtnp1_ytn \f$p(\mathbf{x}_{n+1}\,|\,\mathbf{y}_{1:n})\f$;
   * uncorrected filter density at time \f$t_{n+1}\f$. May be modified.
   * @param flags Optimisation flags.
   *
   * @see Flags for optimisation flags.
   */
  void smooth(const T tn,
      indii::ml::aux::DiracMixturePdf& p_xtn_ytn,
      indii::ml::aux::DiracMixturePdf& p_xtnp1_ytn,
      const unsigned int flags = 0);

  virtual indii::ml::aux::DiracMixturePdf smoothedMeasure();

  /**
   * Get last set of proposal samples.
   *
   * @return Set of proposal samples from last call to smooth().
   */
  indii::ml::aux::DiracMixturePdf& getProposals();
  
  /**
   * Get last set of proposal sample propagations.
   *
   * @return Set of proposal sample propagations from last call to smooth().
   */
  indii::ml::aux::DiracMixturePdf& getPropagations();

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
  KernelForwardBackwardSmootherModel<T>* model;

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
  
};

    }
  }
}

#include "StratifiedParticleResampler.hpp"
#include "../aux/kde.hpp"

#include <assert.h>

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::KernelForwardBackwardSmoother(
    KernelForwardBackwardSmootherModel<T>* model, const NT& N, const KT& K,
    const T tT, const indii::ml::aux::DiracMixturePdf& p_xT) :
    Smoother<T,indii::ml::aux::DiracMixturePdf>(tT, p_xT),
    model(model),
    N(N),
    K(K),
    q_xtns(model->getStateSize()),
    q_xtnp1s(model->getStateSize()) {
  P = p_xT.getDistributedSize(); 
}

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::~KernelForwardBackwardSmoother() {
  //
}

template <class T, class NT, class KT, class ST>
indii::ml::filter::KernelForwardBackwardSmootherModel<T>*
    indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::getModel() {
  return model;
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::smooth(
    const T tn,
    indii::ml::aux::DiracMixturePdf& p_xtn_ytn,
    indii::ml::aux::DiracMixturePdf& p_xtnp1_ytn,
    indii::ml::aux::Pdf& q_xtn,
    const unsigned int flags) {
  const unsigned int D = model->getStateSize();
    
  /* pre-conditions */
  assert (p_xtn_ytn.getDimensions() == D);
  assert (p_xtnp1_ytn.getDimensions() == D);
  assert (q_xtn.getDimensions() == D);

  vector x(D);
  const T del = this->tn - tn;
  DiracMixturePdf& p_xtnp1_ytT = this->p_xtn_ytT;
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

  aux::vector pi(P_local), delta(P_local), psi(P_local);
    
  if (noStandardisation) {
    /* filter density evaluation */
    {
      //p_xtn_ytn.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q_xtns);
      aux::KDTree<ST> targetTree(&p_xtn_ytn);    
      noalias(pi) = aux::distributedDualTreeDensity(queryTree,
          targetTree, p_xtn_ytn.getWeights(), N, K);
    }
    
    /* uncorrected and smooth densities have same support, evaluate
     * together */
    {
      assert (p_xtnp1_ytn.getSize() == p_xtnp1_ytT.getSize());

      aux::KDTree<ST> queryTree(&q_xtnp1s);
      aux::KDTree<ST> targetTree(&p_xtnp1_ytT);

      aux::matrix ws(2,p_xtnp1_ytn.getSize());
      row(ws,0) = p_xtnp1_ytT.getWeights();
      row(ws,1) = p_xtnp1_ytn.getWeights();
      
      aux::matrix result(2,q_xtnp1s.getSize());
      noalias(result) = aux::distributedDualTreeDensity(queryTree,
          targetTree, ws, N, K);
          
      noalias(psi) = row(result,0);
      noalias(delta) = row(result,1);
    }
  } else {
    /* filter density evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      noalias(mu) = p_xtn_ytn.getDistributedExpectation();
      noalias(L) = p_xtn_ytn.getDistributedStandardDeviation();
  
      DiracMixturePdf q(q_xtns);
      q.standardise(mu, L);
      p_xtn_ytn.standardise(mu, L);

      //p_xtn_ytn.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q);
      aux::KDTree<ST> targetTree(&p_xtn_ytn);    
      noalias(pi) = aux::distributedDualTreeDensity(queryTree,
          targetTree, p_xtn_ytn.getWeights(), N, K);
    }
    
    /* smooth density evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      noalias(mu) = p_xtnp1_ytT.getDistributedExpectation();
      noalias(L) = p_xtnp1_ytT.getDistributedStandardDeviation();
  
      DiracMixturePdf q(q_xtnp1s);
      q.standardise(mu, L);
      p_xtnp1_ytT.standardise(mu, L);

      //p_xtnp1_ytT.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q);
      aux::KDTree<ST> targetTree(&p_xtnp1_ytT);
      noalias(psi) = aux::distributedDualTreeDensity(queryTree,
          targetTree, p_xtnp1_ytT.getWeights(), N, K);
    }
    
    /* uncorrected filter density evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      noalias(mu) = p_xtnp1_ytn.getDistributedExpectation();
      noalias(L) = p_xtnp1_ytn.getDistributedStandardDeviation();
  
      DiracMixturePdf q(q_xtnp1s);
      q.standardise(mu, L);
      p_xtnp1_ytn.standardise(mu, L);

      //p_xtnp1_ytn.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q);
      aux::KDTree<ST> targetTree(&p_xtnp1_ytn);
      noalias(delta) = aux::distributedDualTreeDensity(queryTree,
          targetTree, p_xtnp1_ytn.getWeights(), N, K);
    }
  }

  /* build smoothed distribution */
  psi = element_div(element_prod(pi,psi), element_prod(delta,q_ptns));
  this->p_xtn_ytT = q_xtns;
  this->p_xtn_ytT.setWeights(psi);
  
  /* for at least one system, double well, normalisation has proved
   * necessary to prevent degeneracy, and so we include it here. */
  this->p_xtn_ytT.distributedNormalise();
  
  /* update state */
  this->tn = tn;
}

template <class T, class NT, class KT, class ST>
void indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::smooth(
    const T tn,
    indii::ml::aux::DiracMixturePdf& p_xtn_ytn,
    indii::ml::aux::DiracMixturePdf& p_xtnp1_ytn,
    const unsigned int flags) {
  const unsigned int D = model->getStateSize();
    
  /* pre-conditions */
  assert (p_xtn_ytn.getDimensions() == D);
  assert (p_xtnp1_ytn.getDimensions() == D);

  vector x(D);
  const T del = this->tn - tn;
  DiracMixturePdf& p_xtnp1_ytT = this->p_xtn_ytT;
  unsigned int P_local = p_xtn_ytn.getSize();
  unsigned int i;

  /* pick apart flags */
  bool noResampling = flags & NO_RESAMPLING;
  bool noStandardisation = flags & NO_STANDARDISATION;
  bool filterSmoother = flags & FILTER_SMOOTHER;

  q_xtns = p_xtn_ytn;
  if (noResampling && noStandardisation && filterSmoother) {
    /* use filter-smoother */
    q_xtnp1s = p_xtnp1_ytT;
    assert (p_xtn_ytn.getSize() == p_xtnp1_ytT.getSize());  

    aux::vector psi(p_xtnp1_ytT.getWeights());
    this->p_xtn_ytT = p_xtn_ytn;
    this->p_xtn_ytT.setWeights(psi);

    /* update state */
    this->tn = tn;

    return;
  } else if (noResampling) {
    /* use p_xtnp1_ytn particles as propagations */
    q_xtnp1s = p_xtnp1_ytT;
    assert (p_xtn_ytn.getSize() == p_xtnp1_ytT.getSize());  
  } else {
    /* propagate filter particles */
    this->q_delta = del;
    q_xtnp1s.clear();
    for (i = 0; i < p_xtn_ytn.getSize(); i++) {
      noalias(x) = model->transition(p_xtn_ytn.get(i), tn, del);
      q_xtnp1s.add(x, p_xtn_ytn.getWeight(i));
    }
  }

  const aux::vector& pi = p_xtn_ytn.getWeights();
  aux::vector delta(P_local), psi(P_local);
  
  if (noStandardisation) {
    /* uncorrected and smooth densities have same support, evaluate
     * together */
    assert (p_xtnp1_ytn.getSize() == p_xtnp1_ytT.getSize());
    
    if (noResampling) {
      /* query tree also has same support, perform self-tree evaluation */
      aux::KDTree<ST> tree(&p_xtnp1_ytT);

      aux::matrix ws(2,p_xtnp1_ytn.getSize());
      row(ws,0) = p_xtnp1_ytT.getWeights();
      row(ws,1) = p_xtnp1_ytn.getWeights();

      aux::matrix result(2,p_xtnp1_ytn.getSize());
      noalias(result) = aux::distributedSelfTreeDensity(tree, ws, N, K,
          false);

      noalias(psi) = row(result,0);
      noalias(delta) = row(result,1);
    } else {
      /* uncorrected and smooth densities have same support, evaluate
       * together */
      assert (p_xtnp1_ytn.getSize() == p_xtnp1_ytT.getSize());

      //p_xtnp1_ytT.redistributeBySpace();
      //p_xtnp1_ytn.redistributeBySpace();
      aux::KDTree<ST> queryTree(&q_xtnp1s);
      aux::KDTree<ST> targetTree(&p_xtnp1_ytT);

      aux::matrix ws(2,p_xtnp1_ytn.getSize());
      row(ws,0) = p_xtnp1_ytT.getWeights();
      row(ws,1) = p_xtnp1_ytn.getWeights();
      
      aux::matrix result(2,q_xtnp1s.getSize());
      noalias(result) = aux::distributedDualTreeDensity(queryTree,
          targetTree, ws, N, K);

      noalias(psi) = row(result,0);
      noalias(delta) = row(result,1);
    }
  } else {
    /* smooth density evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      noalias(mu) = p_xtnp1_ytT.getDistributedExpectation();
      noalias(L) = p_xtnp1_ytT.getDistributedStandardDeviation();
      p_xtnp1_ytT.standardise(mu, L);
      //p_xtnp1_ytT.redistributeBySpace();

      if (noResampling) {
        /* query tree has same support, perform self-tree evaluation */
        aux::KDTree<ST> tree(&p_xtnp1_ytT);
        noalias(delta) = aux::distributedSelfTreeDensity(tree,
            p_xtnp1_ytT.getWeights(), N, K);
      } else {
        DiracMixturePdf q(q_xtnp1s);
        q.standardise(mu, L);

        aux::KDTree<ST> queryTree(&q);
        aux::KDTree<ST> targetTree(&p_xtnp1_ytT);
        noalias(delta) = aux::distributedDualTreeDensity(queryTree,
            targetTree, p_xtnp1_ytT.getWeights(), N, K);
      }
    }
  
    /* uncorrected filter density evaluation */
    {
      aux::vector mu(D);
      aux::lower_triangular_matrix L(D,D);
      noalias(mu) = p_xtnp1_ytn.getDistributedExpectation();
      noalias(L) = p_xtnp1_ytn.getDistributedStandardDeviation();
      p_xtnp1_ytn.standardise(mu, L);
      //p_xtnp1_ytn.redistributeBySpace();
  
      if (noResampling) {
        /* query tree has same support, perform self-tree evaluation */
        aux::KDTree<ST> tree(&p_xtnp1_ytn);
        noalias(delta) = aux::distributedSelfTreeDensity(tree,
            p_xtnp1_ytn.getWeights(), N, K);
      } else {
        DiracMixturePdf q(q_xtnp1s);
        q.standardise(mu, L);

        aux::KDTree<ST> queryTree(&q);
        aux::KDTree<ST> targetTree(&p_xtnp1_ytn);
        noalias(delta) = aux::distributedDualTreeDensity(queryTree,
            targetTree, p_xtnp1_ytn.getWeights(), N, K);
      }
    }
  }
  
  /* build smoothed distribution */
  psi = element_div(element_prod(pi,psi), delta);
  this->p_xtn_ytT = p_xtn_ytn;
  this->p_xtn_ytT.setWeights(psi);

  /* for at least one system, double well, normalisation has proved
   * necessary to prevent degeneracy, and so we include it here. */
  this->p_xtn_ytT.distributedNormalise();
  
  /* update state */
  this->tn = tn;
}

template <class T, class NT, class KT, class ST>
inline indii::ml::aux::DiracMixturePdf&
    indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::getProposals() {
  return q_xtns;
}

template <class T, class NT, class KT, class ST>
inline indii::ml::aux::DiracMixturePdf&
    indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::getPropagations() {
  return q_xtnp1s;
}

template <class T, class NT, class KT, class ST>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::smoothedMeasure() {
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
void indii::ml::filter::KernelForwardBackwardSmoother<T,NT,KT,ST>::smoothedResample(
    ParticleResampler* resampler) {
  indii::ml::aux::DiracMixturePdf resampled(resampler->resample(
      this->getSmoothedState()));
  this->setSmoothedState(resampled);
}

#endif
