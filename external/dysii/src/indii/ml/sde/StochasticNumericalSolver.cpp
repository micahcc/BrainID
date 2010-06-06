#include "StochasticNumericalSolver.hpp"

using namespace indii::ml::sde;

namespace aux = indii::ml::aux;

StochasticNumericalSolver::StochasticNumericalSolver(
    const unsigned int dimensions, const unsigned int noises) :
    NumericalSolver(dimensions), W(noises) {
  //
}

StochasticNumericalSolver::StochasticNumericalSolver(const aux::vector& y0,
    const unsigned int noises) : NumericalSolver(y0), W(noises) {
  //
}

StochasticNumericalSolver::~StochasticNumericalSolver() {
  //
}

bool StochasticNumericalSolver::sampleNoise(double* ts) {
  /* pre-condition */
  //assert (*ts > t);
  bool sampled;

  if (tf.empty()) {
    /* no future path, sample as normal */
    tf.push(*ts);
    dWf.push(W.sample(*ts - t));
    sampled = true;
  } else {
    if (*ts >= tf.top()) {
      *ts = tf.top();
      sampled = false;
    } else {
      /* future path available, sample Wiener conditioned on this */
      aux::vector dWts((*ts-t)/(tf.top()-t)*dWf.top() + W.sample(*ts-t));
      noalias(dWf.top()) -= dWts;

      tf.push(*ts);
      dWf.push(dWts);
      sampled = true;
    }
  }
  
  /* post-condition */
  assert (!tf.empty());
  assert (!dWf.empty());
  assert (tf.size() == dWf.size());
  assert (tf.top() == *ts);
  
  return sampled;
}

void StochasticNumericalSolver::reset() {
  NumericalSolver::reset();
  tf = std::stack<double>();
  dWf = std::stack<aux::vector>();
}

