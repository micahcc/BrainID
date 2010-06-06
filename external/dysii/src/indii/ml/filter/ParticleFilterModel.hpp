#ifndef INDII_ML_FILTER_PARTICLEFILTERMODEL_HPP
#define INDII_ML_FILTER_PARTICLEFILTERMODEL_HPP

#include "../aux/vector.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * ParticleFilter compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int>
class ParticleFilterModel {
public:
  /**
   * Destructor.
   */
  virtual ~ParticleFilterModel() = 0;

  /**
   * Get number of dimensions in state.
   *
   * @return Number of dimensions in state.
   */
  virtual unsigned int getStateSize() const = 0;

  /**
   * Get number of dimensions in measurements.
   *
   * @return Number of dimensions in measurements.
   */
  virtual unsigned int getMeasurementSize() const = 0;

  /**
   * Propagate particle through the state transition function.
   *
   * @param s \f$\mathbf{s}\f$; a particle.
   * @param t \f$t\f$; start time. This is provided to allow the
   * calculation of deterministic input functions.
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$f(\mathbf{s}, \mathbf{w}, \Delta t)\f$; propagation of
   * the particle through the transition function, with noise.
   */
  virtual int transition(indii::ml::aux::vector& s, const T t, const T delta) const = 0;

  /**
   * Apply the measurement function to an individual particle.
   *
   * @param s \f$\mathbf{s}\f$; a particle.
   *
   * @return \f$g(\mathbf{s},\mathbf{v})\f$; predicted measurement
   * from the particle, with noise.
   */
  virtual indii::ml::aux::vector measure(const indii::ml::aux::vector& s) const = 0;

  /**
   * Weight particle with measurement. The distribution over predicted
   * measurements from the given particle is calculated. The density
   * of this distribution at the actual measurement given becomes the
   * weight assigned to the particle.
   *
   * @param s \f$\mathbf{s}\f$; a particle.
   * @param y \f$\mathbf{y}\f$; the actual measurement.
   *
   * @return Weight assigned to the particle.
   */
  virtual double weight(const indii::ml::aux::vector& s,
      const indii::ml::aux::vector& y) const = 0;

};

    }
  }
}

template <class T>
indii::ml::filter::ParticleFilterModel<T>::~ParticleFilterModel() {
  //
}

#endif

