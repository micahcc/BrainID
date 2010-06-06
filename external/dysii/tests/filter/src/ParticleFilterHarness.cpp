#include "indii/ml/filter/ParticleFilter.hpp"
#include "indii/ml/filter/StratifiedParticleResampler.hpp"
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/aux/vector.hpp"
#include "indii/ml/aux/matrix.hpp"

#include "MobileRobotParticleFilterModel.hpp"
#include "MobileRobot.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

#define SYSTEM_SIZE 2
#define MEAS_SIZE 1
#define ACTUAL_SIZE 3
#define STEPS 250
#define NUM_PARTICLES 1000

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * @file ParticleFilterHarness.cpp
 *
 * Basic test of ParticleFilter.
 *
 * Results are output into files as follows:
 *
 * @section actualPF results/ParticleFilterHarness_actual.out
 *
 * Actual state of the robot at each time. Columns are as follows:
 *
 * @li time
 * @li x coordinate
 * @li y coordinate
 * @li orientation (radians)
 *
 * @section measPF results/ParticleFilterHarness_meas.out
 * 
 * Measurement at each time step. Columns are as follows:
 *
 * @li time
 * @li measurement
 *
 * @section predPF results/ParticleFilterHarness_filter.out
 *
 * Filtered state at each time step. Columns are as follows:
 *
 * @li time
 * @li mean x coordinate
 * @li mean y coordinate
 * @li mean orientation
 * @li The remaining columns give the covariance matrix between the above
 * state variables.
 *
 * @section resultsPF Results
 *
 * Results are as follows:
 *
 * \image html ParticleFilterHarness.png "Results"
 * \image latex ParticleFilterHarness.eps "Results"
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

  /* set up robot simulator */
  MobileRobot robot(0.1, 5e-3);
  
  /* define model */
  MobileRobotParticleFilterModel model(0.1, 5.0e-3);
  aux::GaussianPdf prior(model.suggestPrior());
  aux::DiracMixturePdf x0(prior, NUM_PARTICLES / size);

  /* create filter */
  ParticleFilter<unsigned int> filter(&model, x0);

  /* create resamplers */
  StratifiedParticleResampler resampler(NUM_PARTICLES);

  /* estimate and output results */
  aux::vector meas(MEAS_SIZE);
  aux::vector actual(ACTUAL_SIZE);
  aux::DiracMixturePdf pred(SYSTEM_SIZE);
  unsigned int t = 0;

  ofstream fmeas("results/ParticleFilterHarness_meas.out");
  ofstream factual("results/ParticleFilterHarness_actual.out");
  ofstream fpred("results/ParticleFilterHarness_filter.out");

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

    fpred << t << '\t';
    outputVector(fpred, mu);
    fpred << '\t';
    outputMatrix(fpred, sigma);
    fpred << endl;
  }

  for (t = 1; t <= STEPS; t++) {
    if (rank == 0) {
      robot.move();
      meas = robot.measure();
    }
    boost::mpi::broadcast(world, meas, 0);

    filter.resample(&resampler);
    filter.filter(t, meas);
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
      fpred << t << '\t';
      outputVector(fpred, mu);
      fpred << '\t';
      outputMatrix(fpred, sigma);
      fpred << endl;
    }
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
