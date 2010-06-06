#include "MobileRobotUnscentedKalmanFilterModel.hpp"

#define SYSTEM_SIZE 5
#define SYSTEM_NOISE_SIZE 2
#define MEAS_SIZE 1
#define MEAS_NOISE_SIZE 1

namespace aux = indii::ml::aux;

MobileRobotUnscentedKalmanFilterModel::MobileRobotUnscentedKalmanFilterModel()
    : w(SYSTEM_NOISE_SIZE), v(MEAS_NOISE_SIZE) {
  aux::vector mu;
  aux::symmetric_matrix sigma;

  /* system noise */
  mu.resize(SYSTEM_NOISE_SIZE, false);
  sigma.resize(SYSTEM_NOISE_SIZE, false);

  mu.clear();

  sigma.clear();
  sigma(0,0) = pow(0.01, 2.0);
  sigma(1,1) = pow(0.01, 2.0);

  w.setExpectation(mu);
  w.setCovariance(sigma);

  /* measurement noise */
  mu.resize(MEAS_NOISE_SIZE, false);
  sigma.resize(MEAS_NOISE_SIZE, false);

  mu.clear();

  sigma.clear();
  sigma(0,0) = pow(0.05,2.0);

  v.setExpectation(mu);
  v.setCovariance(sigma);
}

aux::vector MobileRobotUnscentedKalmanFilterModel::transition(
    const aux::vector& x, const aux::vector& w, unsigned int delta) {
  aux::vector xtnp1(SYSTEM_SIZE);
  xtnp1(0) = x(0) + cos(x(2)) * x(3) + w(0);
  xtnp1(1) = x(1) + sin(x(2)) * x(3) + w(1);
  xtnp1(2) = x(2) + x(4);
  xtnp1(3) = x(3);
  xtnp1(4) = x(4);

  return xtnp1;
}

aux::vector MobileRobotUnscentedKalmanFilterModel::measure(
    const aux::vector& x, const aux::vector& v) {
  aux::vector y(MEAS_SIZE);
  y(0) = 2.0 * x(1) + v(0);

  return y;
}

aux::vector MobileRobotUnscentedKalmanFilterModel::backwardTransition(
    const aux::vector& x, const aux::vector& w, unsigned int delta) {
  aux::vector xtnm1(SYSTEM_SIZE);
  xtnm1(0) = x(0) - cos(x(2)) * x(3) - w(0);
  xtnm1(1) = x(1) - sin(x(2)) * x(3) - w(1);
  xtnm1(2) = x(2) - x(4);
  xtnm1(3) = x(3);
  xtnm1(4) = x(4);

  return xtnm1;
}

unsigned int MobileRobotUnscentedKalmanFilterModel::getStateSize() {
  return SYSTEM_SIZE;
}

aux::GaussianPdf& MobileRobotUnscentedKalmanFilterModel::getSystemNoise() {
  return w;
}
   
aux::GaussianPdf& MobileRobotUnscentedKalmanFilterModel::getMeasurementNoise() {
  return v;
}

