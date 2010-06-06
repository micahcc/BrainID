#include "indii/ml/filter/UnscentedKalmanFilter.hpp"
#include "indii/ml/filter/UnscentedKalmanSmoother.hpp"

#include "MobileRobotUnscentedKalmanFilterModel.hpp"
#include "MobileRobot.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <stack>

#define SYSTEM_SIZE 5
#define SYSTEM_NOISE_SIZE 2
#define MEAS_SIZE 1
#define MEAS_NOISE_SIZE 1
#define ACTUAL_SIZE 3
#define STEPS 250

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;
namespace ublas = boost::numeric::ublas;

/**
 * @file UnscentedKalmanSmootherHarness.cpp
 *
 * Basic test of UnscentedKalmanSmoother.
 *
 * Results are output into files as follows:
 *
 * @section actualUKS results/UnscentedKalmanSmootherHarness_actual.out
 *
 * Actual state of the robot at each time. Columns are as follows:
 *
 * @li time
 * @li x coordinate
 * @li y coordinate
 * @li orientation (radians)
 *
 * @section measUKS results/UnscentedKalmanSmootherHarness_meas.out
 * 
 * Measurement at each time step. Columns are as follows:
 *
 * @li time
 * @li measurement
 *
 * @section filterUKS results/UnscentedKalmanSmootherHarness_filter.out
 *
 * Predicted state (filtered) at each time step. Columns are as follows:
 *
 * @li time
 * @li mean x coordinate
 * @li mean y coordinate
 * @li mean orientation
 * @li The remaining columns give the covariance matrix between the above
 * state variables.
 *
 * @section smoothUKS results/UnscentedKalmanSmootherHarness_smooth.out
 *
 * Predicted state (smoothed) at each time step. Columns are as follows:
 *
 * @li time
 * @li mean x coordinate
 * @li mean y coordinate
 * @li mean orientation
 * @li The remaining columns give the covariance matrix between the above
 * state variables.
 *
 * Note that as the smoothing is performed in a backwards pass, this file has
 * entries in reverse time order.
 *
 * @section resultsUKS Results
 *
 * Results are as follows:
 *
 * \image html UnscentedKalmanSmootherHarness.png "Results"
 * \image latex UnscentedKalmanSmootherHarness.eps "Results"
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

  /* create smoother */
  UnscentedKalmanFilter<unsigned int> filter(&model, x0);

  /* estimate and output results */
  stack<aux::GaussianPdf> filteredStates;
  stack<aux::vector> outputs;
  aux::vector meas(MEAS_SIZE);
  aux::vector actual(ACTUAL_SIZE);
  aux::GaussianPdf pred(SYSTEM_SIZE);
  unsigned int t = 0;

  ofstream fmeas("results/UnscentedKalmanSmootherHarness_meas.out");
  ofstream factual("results/UnscentedKalmanSmootherHarness_actual.out");
  ofstream ffilter("results/UnscentedKalmanSmootherHarness_filter.out");
  ofstream fbackfilter(
      "results/UnscentedKalmanSmootherHarness_backfilter.out");
  ofstream fsmooth("results/UnscentedKalmanSmootherHarness_smooth.out");

  /* output initial state */
  actual = robot.getState();
  pred = filter.getFilteredState();
  filteredStates.push(pred);

  cerr << t << ' ';

  factual << t << '\t';
  outputVector(factual, actual);
  factual << endl;

  ffilter << t << '\t';
  outputVector(ffilter, pred.getExpectation());
  ffilter << '\t';
  outputMatrix(ffilter, pred.getCovariance());
  ffilter << endl;

  /* filter */
  for (t = 1; t <= STEPS; t++) {
    robot.move();

    meas = robot.measure();
    filter.filter(t, meas);
    pred = filter.getFilteredState();
    actual = robot.getState();
    outputs.push(meas);
    filteredStates.push(pred);

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
    ffilter << t << '\t';
    outputVector(ffilter, pred.getExpectation());
    ffilter << '\t';
    outputMatrix(ffilter, pred.getCovariance());
    ffilter << endl;
  }

  /* smooth */
  t--;
  pred = filter.getFilteredState();
  UnscentedKalmanSmoother<unsigned int> smoother(&model, t, pred);

  cerr << t << ' ';
  fsmooth << t << '\t';
  outputVector(fsmooth, pred.getExpectation());
  fsmooth << '\t';
  outputMatrix(fsmooth, pred.getCovariance());
  fsmooth << endl;

  pred = smoother.getBackwardFilteredState();
  fbackfilter << t << '\t';
  outputVector(fbackfilter, pred.getExpectation());
  fbackfilter << '\t';
  outputMatrix(fbackfilter, pred.getCovariance());
  fbackfilter << endl;

  outputs.pop();
  filteredStates.pop();

  for (t = STEPS - 1; t >= 1; t--) {
    smoother.smooth(t, outputs.top(), filteredStates.top());
    outputs.pop();
    filteredStates.pop();

    cerr << t << ' ';

    /* output backward filtered state */
    pred = smoother.getBackwardFilteredState();
    fbackfilter << t << '\t';
    outputVector(fbackfilter, pred.getExpectation());
    fbackfilter << '\t';
    outputMatrix(fbackfilter, pred.getCovariance());
    fbackfilter << endl;

    /* output smoothed state */
    pred = smoother.getSmoothedState();
    fsmooth << t << '\t';
    outputVector(fsmooth, pred.getExpectation());
    fsmooth << '\t';
    outputMatrix(fsmooth, pred.getCovariance());
    fsmooth << endl;
  }

  fmeas.close();
  factual.close();
  ffilter.close();
  fbackfilter.close();
  fsmooth.close();

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
