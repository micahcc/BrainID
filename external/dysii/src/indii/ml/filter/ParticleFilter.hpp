#ifndef INDII_ML_FILTER_PARTICLEFILTER_HPP
#define INDII_ML_FILTER_PARTICLEFILTER_HPP

#include "../aux/DiracMixturePdf.hpp"
#include "../aux/parallel.hpp"  

#include "Filter.hpp"
#include "ParticleResampler.hpp"
#include "ParticleFilterModel.hpp"
#include "StratifiedParticleResampler.hpp"
#include "boost/serialization/base_object.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Particle %filter.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * ParticleFilter is suitable for systems with nonlinear transition
 * and measurement functions, approximating state and noise with
 * indii::ml::aux::DiracMixturePdf distributions.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template<class T = unsigned int>
class ParticleFilter : public Filter<T,indii::ml::aux::DiracMixturePdf> {
public:
  /**
   * Create new particle %filter.
   *
   * @param model Model to estimate.
   * @param p_x0 \f$p(\mathbf{x}_0)\f$; prior over the
   * initial state \f$\mathbf{x}_0\f$.
   */
  ParticleFilter(ParticleFilterModel<T>* model,
      indii::ml::aux::DiracMixturePdf& p_x0);

  /**
   * Destructor.
   */
  virtual ~ParticleFilter();

  /**
   * Get the model being estimated.
   *
   * @return The model being estimated.
   */
  virtual ParticleFilterModel<T>* getModel();

  /**
   * Advance system.
   *
   * @param tnp1 \f$t_{n+1}\f$; the time to which to advance the
   * system. This must be greater than the current time \f$t_n\f$.
   */
  virtual void filter(const T tnp1);
  
  /**
   * Update with measurement.
   *
   * @param ytn \f$\mathbf{y}(t_n)\f$; measurement at the current time.
   */
  virtual void filter(const indii::ml::aux::vector& ytn);

  virtual void filter(const T tnp1, const indii::ml::aux::vector& ytnp1);
  
  virtual indii::ml::aux::DiracMixturePdf measure();

  /**
   * Resample the filtered state.
   *
   * @param resampler A particle resampler.
   *
   * Resamples the filtered state using the given
   * ParticleResampler. Zero or more resample() calls may be made
   * between each call of filter() to resample the filtered state as
   * desired.
   *
   * Calling this method and passing in the resampler (visitor
   * pattern) is preferred to separate calls to getFilteredState(),
   * ParticleResampler::resample() and setFilteredState().
   */
  void resample(ParticleResampler* resampler);

protected:
  /**
   * Model.
   */
  ParticleFilterModel<T>* model;

  /**
   * Serialization
   */
  template< class Archive >
  void serialize(Archive & ar, unsigned int version)
  {
    ar & boost::serialization::base_object
                <Filter<T, indii::ml::aux::DiracMixturePdf> >(*this);
//    ar & model;
  }
  friend class boost::serialization::access;

};

    }
  }
}

#include <assert.h>
#include <vector>

template <class T>
indii::ml::filter::ParticleFilter<T>::ParticleFilter(
    ParticleFilterModel<T>* model,
    indii::ml::aux::DiracMixturePdf& p_x0) :
    Filter<T,indii::ml::aux::DiracMixturePdf>(p_x0), model(model) 
{
  //
}

template <class T>
indii::ml::filter::ParticleFilter<T>::~ParticleFilter() 
{
  //
}

template <class T>
indii::ml::filter::ParticleFilterModel<T>*
    indii::ml::filter::ParticleFilter<T>::getModel() 
{
  return model;
}

template <class T>
void indii::ml::filter::ParticleFilter<T>::filter(const T tnp1) 
{
  namespace aux = indii::ml::aux;

  /* pre-condition */
  assert (tnp1 >= this->tn);

  T delta = tnp1 - this->tn;

  /* filter */
  for (unsigned int i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    if(model->transition(this->p_xtn_ytn.get(i), this->tn, delta) < 0) {
        this->p_xtn_ytn.setWeight(i, 0);
      }
  }

  /* Make the distribution dirty since values changed 
   * and force an update of cumulative weights, 
   */
  this->p_xtn_ytn.dirty();

  /* update state */
  this->tn = tnp1;
}

template <class T>
void indii::ml::filter::ParticleFilter<T>::filter(const aux::vector& ytn) 
{
  unsigned int i;

  /* Modify weights */
  aux::vector weights = this->p_xtn_ytn.getWeights();

  /* filter */
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    weights[i] = this->p_xtn_ytn.getWeight(i) * 
          model->weight(this->p_xtn_ytn.get(i), ytn);
  }

  /* Update Weights */
  this->p_xtn_ytn.setWeights(weights);
  this->p_xtn_ytn.dirty();
}

template <class T>
void indii::ml::filter::ParticleFilter<T>::filter(const T tnp1,
    const aux::vector& ytnp1) 
{
  namespace aux = indii::ml::aux;
  
  /* pre-condition */
  assert (tnp1 >= this->tn);

  T delta = tnp1 - this->tn;
  
  /* Modify weights */
  aux::vector weights = this->p_xtn_ytn.getWeights();

  /* filter */
  for (unsigned int i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    //move particle forward, and in case of error make the weight 0
    if(model->transition(this->p_xtn_ytn.get(i), this->tn, delta) < 0)
      weights[i] = 0;
    else //weight particle based on sampling and previous weight
      weights[i] = this->p_xtn_ytn.getWeight(i) * 
            model->weight(this->p_xtn_ytn.get(i), ytnp1);
  }

  /* update state */
  this->tn = tnp1;
  
  /* Update Weights */
  this->p_xtn_ytn.setWeights(weights);
  this->p_xtn_ytn.dirty();
}

template <class T>
indii::ml::aux::DiracMixturePdf indii::ml::filter::ParticleFilter<T>::measure() 
{
  namespace aux = indii::ml::aux;
  unsigned int i;
  aux::DiracMixturePdf p_ytn_xtn(model->getMeasurementSize());  
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    p_ytn_xtn.add(model->measure(this->p_xtn_ytn.get(i)));
  }

  return p_ytn_xtn;
}

template <class T>
void indii::ml::filter::ParticleFilter<T>::resample(ParticleResampler* resampler) 
{
  this->setFilteredState( resampler->resample(this->getFilteredState()) );
}

#endif

