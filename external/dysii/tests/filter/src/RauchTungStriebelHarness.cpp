#include "indii/ml/filter/KalmanFilter.hpp"
#include "indii/ml/filter/RauchTungStriebelSmoother.hpp"
#include "indii/ml/filter/LinearModel.hpp"

#include "MobileRobot.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <stack>

#define STATE_SIZE 5
#define MEAS_SIZE 1
#define ACTUAL_SIZE 3
#define STEPS 100

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;
namespace ublas = boost::numeric::ublas;

/**
 * @file RauchTungStriebelHarness.cpp
 *
 * Basic test of RauchTungStriebelSmoother.
 *
 * This test sets up a linear model for testing the
 * Rauch-Tung-Striebel (RTS) smoother implementation. It may be
 * validated against the Kalman filter and kalman smoother results.
 *
 * Results are output into files as follows:
 *
 * @section actualRTS results/RauchTungStriebelHarness_actual.out
 *
 * Actual state of the robot at each time. Columns are as follows:
 *
 * @li time
 * @li x coordinate
 * @li y coordinate
 * @li orientation (radians)
 *
 * @section measRTS results/RauchTungStriebelHarness_meas.out
 * 
 * Measurement at each time step. Columns are as follows:
 *
 * @li time
 * @li measurement
 *
 * @section filterRTS results/RauchTungStriebelHarness_filter.out
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
 * @section smoothRTS results/RauchTungStriebelHarness_smooth.out
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
 * @section resultsRTS Results
 *
 * Results are as follows:
 *
 * \image html RauchTungStriebelHarness.png "Results"
 * \image latex RauchTungStriebelHarness.eps "Results"
 */

void outputVector(ofstream& out, aux::vector vec);

void outputMatrix(ofstream& out, aux::matrix mat);

/**
 * Run tests
 */
int main(int argc, const char* argv) {
  /* define model */
  aux::matrix A(STATE_SIZE,STATE_SIZE);
  aux::matrix G(STATE_SIZE,STATE_SIZE);
  aux::matrix C(MEAS_SIZE,STATE_SIZE);
  aux::symmetric_matrix Q(STATE_SIZE);
  aux::symmetric_matrix R(MEAS_SIZE);

  A.clear();
  A(0,0) = 1.0;
  A(0,3) = cos(0.8);
  A(1,1) = 1.0;
  A(1,3) = sin(0.8);
  A(2,2) = 1.0;
  A(3,3) = 1.0;
  A(4,4) = 1.0;

  G.clear();
  G(0,0) = 1.0;
  G(1,1) = 1.0;

  Q.clear();
  Q(0,0) = pow(0.01, 2.0);
  Q(1,1) = pow(0.01, 2.0);
  /* next three just so that Q is Cholesky decomposible, are zeroed by G */
  Q(2,2) = 1.0;
  Q(3,3) = 1.0;
  Q(4,4) = 1.0;

  C.clear();
  C(0,1) = 2.0;

  R.clear();
  R(0,0) = pow(0.05,2.0);
  
  LinearModel model(A, G, Q, C, R);

  /* initial state */
  aux::vector mu(STATE_SIZE);
  aux::symmetric_matrix sigma(STATE_SIZE);

  mu.clear();
  mu(0) = -1.0;
  mu(1) = 1.0;
  mu(2) = 0.8;
  mu(3) = 0.1;
  mu(4) = 0.0;

  sigma.clear();
  sigma(0,0) = 1.0;
  sigma(1,1) = 1.0;
  sigma(2,2) = 0.1;
  sigma(3,3) = 0.1;
  sigma(4,4) = 0.1;

  aux::GaussianPdf x0(mu, sigma);

  /* create smoother */
  KalmanFilter<unsigned int> filter(&model, x0);

  /* set up robot simulator */
  MobileRobot robot(0.1, 5e-3);

  /* estimate and output results */
  stack<aux::GaussianPdf> filteredStates;
  aux::vector meas(MEAS_SIZE);
  aux::vector actual(ACTUAL_SIZE);
  aux::GaussianPdf pred(STATE_SIZE);
  unsigned int t = 0;

  ofstream fmeas("results/RauchTungStriebelHarness_meas.out");
  ofstream factual("results/RauchTungStriebelHarness_actual.out");
  ofstream ffilter("results/RauchTungStriebelHarness_filter.out");
  ofstream fsmooth("results/RauchTungStriebelHarness_smooth.out");

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
    filteredStates.push(pred);
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
    ffilter << t << '\t';
    outputVector(ffilter, pred.getExpectation());
    ffilter << '\t';
    outputMatrix(ffilter, pred.getCovariance());
    ffilter << endl;
  }

  /* smooth */
  t--;
  pred = filter.getFilteredState();
  RauchTungStriebelSmoother<unsigned int> smoother(&model, t, pred);

  cerr << t << ' ';
  fsmooth << t << '\t';
  outputVector(fsmooth, pred.getExpectation());
  fsmooth << '\t';
  outputMatrix(fsmooth, pred.getCovariance());
  fsmooth << endl;

  for (t = STEPS - 1; t >= 1; t--) {
    smoother.smooth(t - 1, filteredStates.top());
    filteredStates.pop();
    pred = smoother.getSmoothedState();

    cerr << t << ' ';

    /* output filtered state */
    fsmooth << t << '\t';
    outputVector(fsmooth, pred.getExpectation());
    fsmooth << '\t';
    outputMatrix(fsmooth, pred.getCovariance());
    fsmooth << endl;
  }

  fmeas.close();
  factual.close();
  ffilter.close();
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
