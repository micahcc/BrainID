#ifndef INDII_ML_FILTER_AUXILIARYPARTICLERESAMPLER_HPP
#define INDII_ML_FILTER_AUXILIARYPARTICLERESAMPLER_HPP

#include "ParticleResampler.hpp"
#include "ParticleFilter.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Auxiliary particle resampler.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * Produces a new approximation of a weighted sample set at time
 * \f$t_n\f$ by performing a one step lookahead, favouring those
 * particles more likely to have higher weights at time
 * \f$t_{n+1}\f$. Combined with ParticleFilter, this produces the
 * Auxiliary Particle Filter.
 *
 * @section AuxiliaryParticleResampler_references References
 */
template <class T = unsigned int>
class AuxiliaryParticleResampler : public ParticleResampler {
public:
  /**
   * Constructor.
   *
   * @param filter Particle %filter to which to couple the resampler.
   * @param P Number of particles to resample from each distribution
   * supplied to resample(). If zero, will resample the same number of
   * particles as the distribution supplied.
   */
  AuxiliaryParticleResampler(ParticleFilter<T>* filter,
      const unsigned int P = 0);

  /**
   * Destructor.
   */
  virtual ~AuxiliaryParticleResampler();

  /**
   * Set the number of particles to resample.
   *
   * @param P Number of particles to resample from each distribution
   * supplied to resample(). If zero, will resample the same number of
   * particles as the distribution supplied.
   */
  void setNumParticles(const unsigned int P = 0);

  /**
   * Set lookahead parameters for next resampling.
   *
   * @param tnp1 \f$t_{n+1}\f$; the time at which the next measurement
   * will be available.
   * @param ytnp1 \f$\mathbf{y}(t_{n+1})\f$; the next measurement, at
   * time \f$t_{n+1}\f$.
   *
   * Sets the parameters for the next call to
   * ParticleFilter::filter(). These provide a lookahead to the future
   * time \f$t_{n+1}\f$ on which to base the resampling of particles
   * at the current time \f$t_b\f$.
   */
  void setLookAhead(const T tnp1, const indii::ml::aux::vector& ytnp1);

  virtual indii::ml::aux::DiracMixturePdf resample(
      indii::ml::aux::DiracMixturePdf& p);

private:
  /**
   * Reweighted component.
   */
  struct reweighted_component {
    /**
     * Weight.
     */
    double w;

    /**
     * Reweighted weight.
     */
    double rw;

    /**
     * Component
     */
    indii::ml::aux::DiracPdf x;

    /**
     * Comparison operator for sorting.
     *
     * @return True if this component has greater reweight than the
     * argument's component.
     */
    bool operator<(const reweighted_component& o) const;

  };

  /**
   * Reweighted component vector.
   */
  typedef std::vector<reweighted_component> reweighted_component_vector;

  /**
   * Reweighted component iterator.
   */
  typedef typename reweighted_component_vector::iterator
      reweighted_component_iterator;

  /**
   * Reweighted component const iterator.
   */
  typedef typename reweighted_component_vector::const_iterator
      reweighted_component_const_iterator;

  /**
   * Particle filter to which the resampler is coupled.
   */
  ParticleFilter<T>* filter;

  /**
   * Number of particles to resample from each distribution.
   */
  unsigned int P;

  /**
   * Time at which the next measurement will be available.
   */
  T tnp1;

  /**
   * The next measurement.
   */
  indii::ml::aux::vector ytnp1;

};

    }
  }
}

template <class T>
indii::ml::filter::AuxiliaryParticleResampler<T>::AuxiliaryParticleResampler(
    ParticleFilter<T>* filter, const unsigned int P) : filter(filter), P(P) {
  //
}

template <class T>
indii::ml::filter::AuxiliaryParticleResampler<T>::~AuxiliaryParticleResampler() {
  //
}

template <class T>
void indii::ml::filter::AuxiliaryParticleResampler<T>::setNumParticles(
    const unsigned int P) {
  this->P = P;
}

template <class T>
void indii::ml::filter::AuxiliaryParticleResampler<T>::setLookAhead(
    const T tnp1, const indii::ml::aux::vector& ytnp1) {
  this->tnp1 = tnp1;
  this->ytnp1.resize(ytnp1.size());
  this->ytnp1 = ytnp1;
}

template <class T>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::AuxiliaryParticleResampler<T>::resample(
    indii::ml::aux::DiracMixturePdf& p) {
  namespace aux = indii::ml::aux;

  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

  aux::DiracMixturePdf resampled(p.getDimensions());
  aux::vector x(p.getDimensions());

  ParticleFilterModel<T>* model = filter->getModel();
  const T tn = filter->getTime();
  const T delta = tnp1 - tn;

  unsigned int i;
  unsigned int P = this->P;
  if (P == 0) {
    P = p.getDistributedSize();
  }
  const unsigned int P_local = p.getSize();

  double W_rw;  // local reweight total
  double W_rws; // scan sum reweight
  double W_rwt; // distributed reweight total
  aux::vector rw(P_local); // auxiliary weights

  /* calculate auxiliary weights */
  W_rw = 0.0;
  for (i = 0; i < P_local; i++) {
    noalias(x) = model->transition(p.get(i), tn, delta);
    rw(i) = model->weight(x, ytnp1) * p.getWeight(i);
    W_rw += rw(i);
  }

  /* scan sum and total reweights */
  W_rws = boost::mpi::scan(world, W_rw, std::plus<double>());
  if (rank == size - 1) {
    W_rwt = W_rws; // already has total weight
  }
  boost::mpi::broadcast(world, W_rwt, size - 1);

  /* generate common random alpha across nodes */
  double alpha;
  if (rank == 0) {
    alpha = aux::Random::uniform(0.0, 1.0);
  }
  boost::mpi::broadcast(world, alpha, 0);

  /* resample */
  double u = 0.0;
  double w, j, rem;

  w = W_rwt / P;
  rem = fmod(W_rws - W_rw, w);
  if (rem >= alpha*w) {
    j = 1.0 + alpha - rem/w;
  } else {
    j = alpha - rem/w;
  }

  for (i = 0; i < P_local; i++) {
    u += rw(i);
    while (u >= w * j) {
      resampled.add(p.get(i), p.getWeight(i) / rw(i));
      j += 1.0;
    }
  }

  resampled.redistributeBySize();
  
  return resampled;
}

template <class T>
bool indii::ml::filter::AuxiliaryParticleResampler<T>::reweighted_component::operator<(
    const reweighted_component& o) const {
  return this->rw > o.rw;
}

#endif
