#ifndef MOBILEROBOT_HPP
#define MOBILEROBOT_HPP

#include "indii/ml/aux/GaussianPdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"

namespace aux = indii::ml::aux;

class MobileRobot {
public:
  /**
   * Create new mobile robot.
   *
   * @param v Translational velocity.
   * @param w Angular velocity.
   */
  MobileRobot(double v = 0.1, double w = 0.0);

  /**
   * Destructor.
   */
  virtual ~MobileRobot();

  /**
   * Move the robot to its position at the next time step.
   */
  void move();

  /**
   * Take a measurement from the system.
   */
  aux::vector measure();

  /**
   * Get the current actual state of the robot.
   */
  const aux::vector& getState();

private:
  /**
   * Noise intrinsic to system.
   */
  aux::GaussianPdf systemNoise;

  /**
   * Measurement noise.
   */
  aux::GaussianPdf measNoise;

  /**
   * Measurement model.
   */
  aux::matrix measModel;

  /**
   * Current state:
   *
   * @li x coordinate
   * @li y coordinate
   * @li orientation
   */
  aux::vector state;

  /**
   * Translational velocity.
   */
  const double v;

  /**
   * Angular velocity.
   */
  const double w;

  /**
   * State space size.
   */
  static const unsigned int STATE_SIZE = 3;

  /**
   * Measurement space size.
   */
  static const unsigned int MEAS_SIZE = 1;

};

#endif
