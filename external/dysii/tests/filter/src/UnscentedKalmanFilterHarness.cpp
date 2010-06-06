#include "indii/ml/filter/UnscentedKalmanFilter.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"

#include "MobileRobotUnscentedKalmanFilterModel.hpp"
#include "MobileRobot.hpp"

#include <math.h>
#include <iostream>
#include <fstream>

#define SYSTEM_SIZE 5
#define SYSTEM_NOISE_SIZE 2
#define MEAS_SIZE 1
#define MEAS_NOISE_SIZE 1
#define ACTUAL_SIZE 3
#define STEPS 250

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * @file UnscentedKalmanFilterHarness.cpp
 *
 * Basic test of UnscentedKalmanFilter.
 *
 * Results are output into files as follows:
 *
 * @section actualUKF results/UnscentedKalmanFilterHarness_actual.out
 *
 * Actual state of the robot at each time. Columns are as follows:
 *
 * @li time
 * @li x coordinate
 * @li y coordinate
 * @li orientation (radians)
 *
 * @section measUKF results/UnscentedKalmanFilterHarness_meas.out
 * 
 * Measurement at each time step. Columns are as follows:
 *
 * @li time
 * @li measurement
 *
 * @section predUKF results/UnscentedKalmanFilterHarness_pred.out
 *
 * Predicted state at each time step. Columns are as follows:
 *
 * @li time
 * @li mean x coordinate
 * @li mean y coordinate
 * @li mean orientation
 * @li The remaining columns give the covariance matrix between the above
 * state variables.
 *
 * @section resultsUKF Results
 *
 * Results are as follows:
 *
 * \image html UnscentedKalmanFilterHarness.png "Results, c.f. BFL Tutorial Figures 3.2 and 3.3"
 * \image latex UnscentedKalmanFilterHarness.eps "Results, c.f. BFL Tutorial Figures 3.2 and 3.3"
 */

void outputVector(ofstream& out, aux::vector vec);

void outputMatrix(ofstream& out, aux::matrix mat);

/**
 * Run tests
 */
int main(int argc, const char* argv) {
  /* define model */
  MobileRobotUnscentedKalmanFilterModel model;

  /* set up robot simulator */
  MobileRobot robot(0.1, 5e-3);

  /* initial state */
  aux::vector mu(SYSTEM_SIZE);
  aux::symmetric_matrix sigma(SYSTEM_SIZE);

  mu.clear();
  mu(0) = -1.0;
  mu(1) = 1.0;
  mu(2) = 0.8;
  mu(3) = 0.1;
  mu(4) = 5e-3;

  sigma.clear();
  sigma(0,0) = 1.0;
  sigma(1,1) = 1.0;
  sigma(2,2) = 0.01;
  sigma(3,3) = 1e-6;
  sigma(4,4) = 1e-6;

  aux::GaussianPdf x0(mu, sigma);

  /* create filter */
  UnscentedKalmanFilter<unsigned int> filter(&model, x0);

  /* estimate and output results */
  aux::vector meas(MEAS_SIZE);
  aux::vector actual(ACTUAL_SIZE);
  aux::GaussianPdf pred(SYSTEM_SIZE);
  unsigned int t = 0;

  ofstream fmeas("results/UnscentedKalmanFilterHarness_meas.out");
  ofstream factual("results/UnscentedKalmanFilterHarness_actual.out");
  ofstream fpred("results/UnscentedKalmanFilterHarness_pred.out");

  /* output initial state */
  pred = filter.getFilteredState();
  actual = robot.getState();

  cerr << t << ' ';

  factual << t << '\t';
  outputVector(factual, actual);
  factual << endl;

  fpred << t << '\t';
  outputVector(fpred, pred.getExpectation());
  fpred << '\t';
  outputMatrix(fpred, pred.getCovariance());
  fpred << endl;

  for (t = 1; t <= STEPS; t++) {
      robot.move();
      meas = robot.measure();
      filter.filter(t, meas);
      pred = filter.getFilteredState();
      actual = robot.getState();

      cerr << t << ' ';

      /* output measurement */
      fmeas << t << '\t';
      outputVector(fmeas, meas);
      fmeas << endl;

      /* output actual state */
      factual << t << '\t';
      outputVector(factual, actual);
      factual << endl;

      /* output filtered state */
      fpred << t << '\t';
      outputVector(fpred, pred.getExpectation());
      fpred << '\t';
      outputMatrix(fpred, pred.getCovariance());
      fpred << endl;
  }

  fmeas.close();
  factual.close();
  fpred.close();

  return 0;
}

void outputVector(ofstream& out, aux::vector vec) {
  aux::vector::iterator iter, end;
  iter = vec.begin();
  end = vec.end();
  while (iter != end) {
    out << *iter;
    iter++;
    if (iter != end) {
      out << '\t';
    }
  }
}

void outputMatrix(ofstream& out, aux::matrix mat) {
  unsigned int i, j;
  for (j = 0; j < mat.size2(); j++) {
    for (i = 0; i < mat.size1(); i++) {
      out << mat(i,j);
      if (i != mat.size1() - 1 || j != mat.size2() - 1) {
        out << '\t';
      }
    }
  }
}
