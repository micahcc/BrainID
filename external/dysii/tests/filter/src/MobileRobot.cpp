#include "MobileRobot.hpp"

#include <math.h>

namespace aux = indii::ml::aux;
namespace ublas = boost::numeric::ublas;

MobileRobot::MobileRobot(double v, double w) : systemNoise(STATE_SIZE),
    measNoise(MEAS_SIZE), measModel(MEAS_SIZE,STATE_SIZE), state(STATE_SIZE),
    v(v), w(w) {

  /* system noise */
  aux::vector systemNoiseMu(STATE_SIZE);
  aux::symmetric_matrix systemNoiseSigma(STATE_SIZE);
  systemNoiseMu.clear();
  systemNoiseSigma.clear();
  systemNoiseSigma(0,0) = pow(0.01,2.0);
  systemNoiseSigma(1,1) = pow(0.01,2.0);
  systemNoiseSigma(2,2) = pow(0.01*3.14/180,2.0);

  systemNoise.setExpectation(systemNoiseMu);
  systemNoise.setCovariance(systemNoiseSigma);

  /* measurement noise */
  aux::vector measNoiseMu(MEAS_SIZE);
  aux::symmetric_matrix measNoiseSigma(MEAS_SIZE);

  measNoiseMu.clear();
  measNoiseSigma.clear();
  measNoiseSigma(0,0) = pow(0.05,2.0);

  measNoise.setExpectation(measNoiseMu);
  measNoise.setCovariance(measNoiseSigma);

  /* measurement model */
  measModel.clear();
  measModel(0,1) = 2.0;

  /* initial state */
  state.clear();
  state(2) = 0.8;
}

MobileRobot::~MobileRobot() {
  // nothing to do
}

void MobileRobot::move() {
  aux::vector noise(systemNoise.sample());

  state(0) += cos(state(2)) * v + noise(0);
  state(1) += sin(state(2)) * v + noise(1);
  state(2) += w;
}

aux::vector MobileRobot::measure() {
  return (prod(measModel, state) + measNoise.sample());
}

const aux::vector& MobileRobot::getState() {
  return state;
}
