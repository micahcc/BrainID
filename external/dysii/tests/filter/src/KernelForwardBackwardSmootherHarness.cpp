#include "indii/ml/filter/KernelForwardBackwardSmoother.hpp"
#include "indii/ml/filter/ParticleFilter.hpp"
#include "indii/ml/filter/StratifiedParticleResampler.hpp"
#include "indii/ml/aux/Random.hpp"

#include "MobileRobotParticleFilterModel.hpp"
#include "MobileRobot.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>

#define SYSTEM_SIZE 2
#define MEAS_SIZE 1
#define ACTUAL_SIZE 3

#define T 250
#define P 250
#define H aux::hopt(SYSTEM_SIZE, P)

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * @file KernelForwardBackwardSmootherHarness.cpp
 *
 * Basic test of KernelForwardBackwardSmoother.
 *
 * Results are output into files as follows:
 *
 * @section actualPS results/KernelForwardBackwardSmootherHarness_actual.out
 *
 * Actual state of the robot at each time. Columns are as follows:
 *
 * @li time
 * @li x coordinate
 * @li y coordinate
 * @li orientation (radians)
 *
 * @section measPS results/KernelForwardBackwardSmootherHarness_meas.out
 * 
 * Measurement at each time step. Columns are as follows:
 *
 * @li time
 * @li measurement
 *
 * @section filterPS results/KernelForwardBackwardSmootherHarness_filter.out
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
 * @section smoothPS results/KernelForwardBackwardSmootherHarness_smooth.out
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
 * @section resultsPS Results
 *
 * Results are as follows:
 *
 * \image html KernelForwardBackwardSmootherHarness.png "Results"
 * \image latex KernelForwardBackwardSmootherHarness.eps "Results"
 */

void outputVector(ofstream& out, aux::vector vec);

void outputMatrix(ofstream& out, aux::matrix mat);

/**
 * Run tests
 */
int main(int argc, char* argv[]) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

  aux::Random::seed(rank);

  /* set up robot simulator */
  MobileRobot robot(0.1, 5e-3);

  /* define model */
  MobileRobotParticleFilterModel model(0.1, 5e-3);
  aux::GaussianPdf prior(model.suggestPrior());
  aux::DiracMixturePdf x0(prior, P / size);

  /* create filter */
  ParticleFilter<> filter(&model, x0);
  StratifiedParticleResampler resampler(P);

  /* estimate and output results */
  stack<aux::DiracMixturePdf> filteredStates;
  stack<aux::DiracMixturePdf> incrementFilteredStates;
  stack<bool> resamples;

  aux::vector meas(MEAS_SIZE);
  aux::vector actual(ACTUAL_SIZE);
  aux::DiracMixturePdf pred(SYSTEM_SIZE);
  unsigned int t = 0;

  ofstream fmeas("results/KernelForwardBackwardSmootherHarness_meas.out");
  ofstream factual("results/KernelForwardBackwardSmootherHarness_actual.out");
  ofstream ffilter("results/KernelForwardBackwardSmootherHarness_filter.out");
  ofstream fsmooth("results/KernelForwardBackwardSmootherHarness_smooth.out");

  aux::vector mu(SYSTEM_SIZE);
  aux::symmetric_matrix sigma(SYSTEM_SIZE);

  /* output initial state */
  pred = filter.getFilteredState();
  actual = robot.getState();
  mu = pred.getDistributedExpectation();
  sigma = pred.getDistributedCovariance();

  if (rank == 0) {
    cerr << t << ' ';

    factual << t << '\t';
    outputVector(factual, actual);
    factual << endl;

    ffilter << t << '\t';
    outputVector(ffilter, mu);
    ffilter << '\t';
    outputMatrix(ffilter, sigma);
    ffilter << endl;
  }

  for (t = 1; t <= T; t++) {
    if (rank == 0) {
      robot.move();
      meas = robot.measure();
    }
    boost::mpi::broadcast(world, meas, 0);

    if (filter.getFilteredState().calculateDistributedEss() < 0.8*P) {
      filter.resample(&resampler);
      resamples.push(true);
    } else {
      resamples.push(false);
    }

    filter.filter(t);
    incrementFilteredStates.push(filter.getFilteredState());
    filter.filter(meas);
    filteredStates.push(filter.getFilteredState());

    pred = filter.getFilteredState();    
    actual = robot.getState();
    mu = pred.getDistributedExpectation();
    sigma = pred.getDistributedCovariance();

    if (rank == 0) {
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
      outputVector(ffilter, mu);
      ffilter << '\t';
      outputMatrix(ffilter, sigma);
      ffilter << endl;
    }
  }

  /* smooth */
  t--;
  pred = filter.getFilteredState();
  
  aux::Almost2Norm N;
  aux::AlmostGaussianKernel K(SYSTEM_SIZE, H);
  KernelForwardBackwardSmoother<> smoother(&model, N, K, t, pred);
  
  mu = pred.getDistributedExpectation();
  sigma = pred.getDistributedCovariance();
  unsigned int flags;

  if (rank == 0) {
    cerr << t << ' ';

    fsmooth << t << '\t';
    outputVector(fsmooth, mu);
    fsmooth << '\t';
    outputMatrix(fsmooth, sigma);
    fsmooth << endl;
  }

  filteredStates.pop();
  
  /* perform other steps */
  for (t = T - 1; t >= 1; t--) {
    /* optimisations */
    flags = NO_STANDARDISATION;
    if (!resamples.top()) {
      flags |= NO_RESAMPLING;
    }

    /* proposal */
    //aux::GaussianPdf q(filteredStates.top().getDistributedExpectation(),
    //    filteredStates.top().getDistributedCovariance());

    smoother.smooth(t, filteredStates.top(), incrementFilteredStates.top(),
        flags);
    
    incrementFilteredStates.pop();
    filteredStates.pop();
    resamples.pop();

    pred = smoother.getSmoothedState();
    mu = pred.getDistributedExpectation();
    sigma = pred.getDistributedCovariance();

    if (rank == 0) {
      cerr << t << ' ';

      /* output smoothed state */
      fsmooth << t << '\t';
      outputVector(fsmooth, mu);
      fsmooth << '\t';
      outputMatrix(fsmooth, sigma);
      fsmooth << endl;
    }
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
