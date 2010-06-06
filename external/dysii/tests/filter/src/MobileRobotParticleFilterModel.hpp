#ifndef MOBILEROBOTPARTICLEFILTERMODEL_HPP
#define MOBILEROBOTPARTICLEFILTERMODEL_HPP

#include "indii/ml/filter/ParticleSmootherModel.hpp"
#include "indii/ml/filter/KernelForwardBackwardSmootherModel.hpp"
#include "indii/ml/filter/KernelTwoFilterSmootherModel.hpp"
#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"

using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * Mobile robot model for particle filter tests.
 */
class MobileRobotParticleFilterModel
    : public virtual ParticleSmootherModel<unsigned int>,
      public virtual KernelForwardBackwardSmootherModel<unsigned int>,
      public virtual KernelTwoFilterSmootherModel<unsigned int> {
public:
  MobileRobotParticleFilterModel(const double vel = 0.1,
      const double ang = 0.0);

  virtual ~MobileRobotParticleFilterModel();

  aux::GaussianPdf suggestPrior();

  virtual unsigned int getStateSize();

  virtual unsigned int getMeasurementSize();

  virtual aux::vector transition(const aux::vector& x,
      const unsigned int start, const unsigned int delta);

  virtual double weight(const aux::vector& x, const aux::vector& y);

  virtual aux::vector measure(const aux::vector& x);

  virtual aux::sparse_matrix alpha(const aux::DiracMixturePdf& p_xtn_ytn,
      const aux::DiracMixturePdf& p_xtnp1_ytnp1, const unsigned int start,
      const unsigned int delta);

private:
  /**
   * System noise.
   */
  aux::GaussianPdf w;

  /**
   * Measurement noise.
   */
  aux::GaussianPdf v;

  /**
   * Translational velocity.
   */
  const double vel;
  
  /**
   * Angular velocity.
   */
  const double ang;
  
};

#endif

