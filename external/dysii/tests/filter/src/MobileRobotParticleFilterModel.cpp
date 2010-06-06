#include "MobileRobotParticleFilterModel.hpp"

#define SYSTEM_SIZE 2
#define MEAS_SIZE 1

MobileRobotParticleFilterModel::MobileRobotParticleFilterModel(
    const double vel, const double ang) :
    w(SYSTEM_SIZE), v(MEAS_SIZE), vel(vel), ang(ang) {
  aux::vector mu(SYSTEM_SIZE);
  aux::symmetric_matrix sigma(SYSTEM_SIZE);
    
  /* system noise */
  mu.clear();

  sigma.clear();
  sigma(0,0) = pow(0.01,2.0);
  sigma(1,1) = pow(0.01,2.0);

  w.setExpectation(mu);
  w.setCovariance(sigma);

  /* measurement noise */
  mu.resize(MEAS_SIZE, false);
  sigma.resize(MEAS_SIZE, false);

  mu.clear();
  sigma.clear();
  sigma(0,0) = pow(0.05, 2.0);

  v.setExpectation(mu);
  v.setCovariance(sigma);
}

MobileRobotParticleFilterModel::~MobileRobotParticleFilterModel() {
  //
}

aux::GaussianPdf MobileRobotParticleFilterModel::suggestPrior() {
  aux::vector mu(SYSTEM_SIZE);
  aux::symmetric_matrix sigma(SYSTEM_SIZE);

  mu.clear();
  mu(0) = -1.0;
  mu(1) = 1.0;

  sigma.clear();
  sigma(0,0) = 1.0;
  sigma(1,1) = 1.0;
  
  aux::GaussianPdf x0(mu, sigma);
  
  return x0;
}

unsigned int MobileRobotParticleFilterModel::getStateSize() {
  return SYSTEM_SIZE;
}

unsigned int MobileRobotParticleFilterModel::getMeasurementSize() {
  return MEAS_SIZE;
}

aux::vector MobileRobotParticleFilterModel::transition(const aux::vector& x,
      const unsigned int start, const unsigned int delta) {
  aux::vector w(SYSTEM_SIZE);
  w = this->w.sample();

  aux::vector xtnp1(SYSTEM_SIZE);
  xtnp1(0) = x(0) + cos(0.8 + ang*start) * vel;
  xtnp1(1) = x(1) + sin(0.8 + ang*start) * vel;

  return (xtnp1 + this->w.sample());
}

double MobileRobotParticleFilterModel::weight(const aux::vector& x,
    const aux::vector& y) {
  aux::vector mu(MEAS_SIZE);
  mu(0) = 2.0 * x(1);

  return v.calculateDensity(y - mu);
}

aux::vector MobileRobotParticleFilterModel::measure(const aux::vector& x) {
  aux::vector y(MEAS_SIZE);
  y(0) = 2.0 * x(1);

  return y + v.sample();
}

aux::sparse_matrix MobileRobotParticleFilterModel::alpha(
    const aux::DiracMixturePdf& p_xtn_ytn,
    const aux::DiracMixturePdf& p_xtnp1_ytnp1,
    const unsigned int start, const unsigned int delta) {
  const unsigned int P1 = p_xtn_ytn.getSize();
  const unsigned int P2 = p_xtnp1_ytnp1.getSize();
  unsigned int i, j;
  double p;
  aux::sparse_matrix alpha(P2,P1);
  aux::vector mu(SYSTEM_SIZE);

  for (i = 0; i < p_xtn_ytn.getSize(); i++) {
    const aux::vector& x = p_xtn_ytn.get(i);
  
    /* propagate particle through transition to get mean */
    mu(0) = x(0) + cos(0.8 + ang*start) * vel;
    mu(1) = x(1) + sin(0.8 + ang*start) * vel;

    for (j = 0; j < P2; j++) {
      p = w.calculateDensity(p_xtnp1_ytnp1.get(j) - mu);
      if (p > 0.0) {
      	alpha(j,i) = p;
      }
    }
  }

  return alpha;
}

