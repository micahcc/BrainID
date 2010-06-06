#ifndef MOBILEROBOTUNSCENTEDKALMANFILTERMODEL_HPP
#define MOBILEROBOTUNSCENTEDKALMANFILTERMODEL_HPP

#include "indii/ml/filter/UnscentedKalmanSmootherModel.hpp"
#include "indii/ml/aux/vector.hpp"

using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * Mobile robot model for unscented Kalman filter and smoother tests.
 */
class MobileRobotUnscentedKalmanFilterModel
    : public UnscentedKalmanSmootherModel<unsigned int> {
public:
  MobileRobotUnscentedKalmanFilterModel();

  virtual unsigned int getStateSize();

  virtual indii::ml::aux::GaussianPdf& getSystemNoise();
   
  virtual indii::ml::aux::GaussianPdf& getMeasurementNoise();

  virtual aux::vector transition(const aux::vector& x, const aux::vector& w,
      unsigned int delta);

  virtual aux::vector measure(const aux::vector& x, const aux::vector& v);

  virtual aux::vector backwardTransition(const aux::vector& x,
      const aux::vector& w, unsigned int delta);

private:
  /**
   * System noise.
   */
  aux::GaussianPdf w;
  
  /**
   * Measurement noise.
   */
  aux::GaussianPdf v;
  
};

#endif
