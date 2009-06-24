#ifndef PARTICLEFILTERMOD_HPP
#define PARTICLEFILTERMOD_HPP

#include <indii/ml/aux/DiracMixturePdf.hpp>
//#include "indii/ml/aux/parallel.hpp"  

#include <indii/ml/filter/Filter.hpp>
#include <indii/ml/filter/ParticleResampler.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>

#include <assert.h>
#include <vector>

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Particle %filter mod.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @modder Micah Chambers <micahc@vt.edu>
 *
 * @param T The type of time.
 *
 * ParticleFilterMod is suitable for systems with nonlinear transition
 * and measurement functions, approximating state and noise with
 * indii::ml::aux::DiracMixturePdf distributions.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */

template<class T = double>
class ParticleFilterMod : public Filter<T,indii::ml::aux::DiracMixturePdf> {
public:
    /**
     * Create new particle %filter.
     *
     * @param model Model to estimate.
     * @param p_x0 \f$p(\mathbf{x}_0)\f$; prior over the
     * initial state \f$\mathbf{x}_0\f$.
     */
    ParticleFilterMod(ParticleFilterModel<T>* model,
            indii::ml::aux::DiracMixturePdf& p_x0);

    /**
     * Destructor.
     */
    virtual ~ParticleFilterMod();

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
    virtual void filter(const indii::ml::aux::vector& input, const T tnp1);

    /**
     * Update with measurement.
     *
     * @param ytn \f$\mathbf{y}(t_n)\f$; measurement at the current time.
     */
    virtual void filter(const indii::ml::aux::vector& ytn);

    virtual void filter(const T tnp1, const indii::ml::aux::vector& ytnp1);
    virtual void filter(const indii::ml::aux::vector& input,
                const T tnp1, const indii::ml::aux::vector& ytnp1);

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

    //don't use this....
    virtual indii::ml::aux::DiracMixturePdf measure();


protected:
    /**
     * Model.
     */
    ParticleFilterModel<T>* model;

};

}}}


template <class T>
indii::ml::filter::ParticleFilterMod<T>::ParticleFilterMod(
    ParticleFilterModel<T>* model,
    indii::ml::aux::DiracMixturePdf& p_x0) :
    Filter<T,indii::ml::aux::DiracMixturePdf>(p_x0), model(model) 
{
  //
}

template <class T>
indii::ml::filter::ParticleFilterMod<T>::~ParticleFilterMod() 
{
  //
}

template <class T>
indii::ml::filter::ParticleFilterModel<T>*
    indii::ml::filter::ParticleFilterMod<T>::getModel() 
{
  return model;
}

template <class T>
void indii::ml::filter::ParticleFilterMod<T>::filter(const T tnp1)
{
  namespace aux = indii::ml::aux;

  /* pre-condition */
  assert (tnp1 >= this->tn);

  aux::DiracMixturePdf p_xtnp1_ytnp1(model->getStateSize());
  aux::vector x(model->getStateSize());
  double w;
  T delta = tnp1 - this->tn;
  unsigned int i;

  /* filter */
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    noalias(x) = model->transition(this->p_xtn_ytn.get(i), this->tn, delta);
    w = this->p_xtn_ytn.getWeight(i);

    p_xtnp1_ytnp1.add(x, w);
  }

  /* update state */
  this->tn = tnp1;
  this->p_xtn_ytn = p_xtnp1_ytnp1;
}

template <class T>
void indii::ml::filter::ParticleFilterMod<T>::filter(
            const indii::ml::aux::vector& input, const T tnp1) 
{
  namespace aux = indii::ml::aux;

  /* pre-condition */
  assert (tnp1 >= this->tn);

  aux::DiracMixturePdf p_xtnp1_ytnp1(model->getStateSize());
  aux::vector x(model->getStateSize());
  double w;
  T delta = tnp1 - this->tn;
  unsigned int i;

  /* filter */
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    noalias(x) = model->transition(this->p_xtn_ytn.get(i), this->tn, delta, input);
    w = this->p_xtn_ytn.getWeight(i);

    p_xtnp1_ytnp1.add(x, w);
  }

  /* update state */
  this->tn = tnp1;
  this->p_xtn_ytn = p_xtnp1_ytnp1;
}

template <class T>
void indii::ml::filter::ParticleFilterMod<T>::filter(const aux::vector& ytn) 
{
  aux::vector x(model->getStateSize());
  aux::vector ws(this->p_xtn_ytn.getSize());
  unsigned int i;
  
  /* filter */
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    noalias(x) = this->p_xtn_ytn.get(i);
    ws(i) = this->p_xtn_ytn.getWeight(i) * model->weight(x, ytn);
  }

  /* update state */
  this->p_xtn_ytn.setWeights(ws);
}

template <class T>
void indii::ml::filter::ParticleFilterMod<T>::filter(const T tnp1,
            const aux::vector& ytnp1) 
{
  namespace aux = indii::ml::aux;
  
  /* pre-condition */
  assert (tnp1 >= this->tn);

  aux::DiracMixturePdf p_xtnp1_ytnp1(model->getStateSize());
  aux::vector x(model->getStateSize());
  double w;
  T delta = tnp1 - this->tn;
  unsigned int i;

  /* filter */
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    noalias(x) = model->transition(this->p_xtn_ytn.get(i), this->tn, delta);
    w = this->p_xtn_ytn.getWeight(i) * model->weight(x, ytnp1);

    p_xtnp1_ytnp1.add(x, w);
  }

  /* update state */
  this->tn = tnp1;
  this->p_xtn_ytn = p_xtnp1_ytnp1;
}

template <class T>
void indii::ml::filter::ParticleFilterMod<T>::filter(
            const indii::ml::aux::vector& input, const T tnp1, 
            const aux::vector& ytnp1)
{
  namespace aux = indii::ml::aux;
  
  /* pre-condition */
  assert (tnp1 >= this->tn);

  aux::DiracMixturePdf p_xtnp1_ytnp1(model->getStateSize());
  aux::vector x(model->getStateSize());
  double w;
  T delta = tnp1 - this->tn;
  unsigned int i;

  /* filter */
  for (i = 0; i < this->p_xtn_ytn.getSize(); i++) {
    noalias(x) = model->transition(this->p_xtn_ytn.get(i), this->tn, delta, input);
    w = this->p_xtn_ytn.getWeight(i) * model->weight(x, ytnp1);

    p_xtnp1_ytnp1.add(x, w);
  }

  /* update state */
  this->tn = tnp1;
  this->p_xtn_ytn = p_xtnp1_ytnp1;
}

template <class T>
void indii::ml::filter::ParticleFilterMod<T>::resample(ParticleResampler* resampler) 
{
  indii::ml::aux::DiracMixturePdf resampled(resampler->resample(
      this->getFilteredState()));
  this->setFilteredState(resampled);
}

template <class T>
indii::ml::aux::DiracMixturePdf
    indii::ml::filter::ParticleFilterMod<T>::measure() {
  namespace aux = indii::ml::aux;

  unsigned int i;
  StratifiedParticleResampler resampler;
  aux::DiracMixturePdf resampled(resampler.resample(
      this->getFilteredState()));

  aux::DiracMixturePdf p_ytn_xtn(model->getMeasurementSize());  
  for (i = 0; i < resampled.getSize(); i++) {
    p_ytn_xtn.add(model->measure(resampled.get(i)));
  }

  return p_ytn_xtn;
}

#endif

