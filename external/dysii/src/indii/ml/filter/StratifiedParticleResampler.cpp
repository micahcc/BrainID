//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "StratifiedParticleResampler.hpp"

namespace aux = indii::ml::aux;

using namespace indii::ml::filter;

StratifiedParticleResampler::StratifiedParticleResampler(
    const unsigned int P) : P(P) {
  //
}

StratifiedParticleResampler::~StratifiedParticleResampler() {
  //
}

void StratifiedParticleResampler::setNumParticles(const unsigned int P) {
  this->P = P;
}

aux::DiracMixturePdf StratifiedParticleResampler::resample(
    aux::DiracMixturePdf& p) {
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

  aux::DiracMixturePdf resampled(p.getDimensions());

  unsigned int P = this->P;
  if (P == 0) {
    P = p.getDistributedSize();
  }

  /* scan sum and total weights */
  double W_s, W;
  W_s = boost::mpi::scan(world, p.getTotalWeight(), std::plus<double>());
  if (rank == size - 1) {
    W = W_s;  // already has total weight
  }
  boost::mpi::broadcast(world, W, size - 1);

  /* generate common random alpha across nodes */
  double alpha;
  if (rank == 0) {
    alpha = aux::Random::uniform(0.0, 1.0);
  }
  boost::mpi::broadcast(world, alpha, 0);

  /* resample */
  unsigned int i;
  double u = 0.0;
  double w, j, rem;

  w = W / P;
  rem = fmod(W_s - p.getTotalWeight(), w);
  if (rem >= alpha*w) {
    j = 1.0 + alpha - rem/w; // previous node samples from first strata
  } else {
    j = alpha - rem/w; // this node samples from first strata
  }

  for (i = 0; i < p.getSize(); i++) {
    u += p.getWeight(i);
    while (u >= w * j) {
      resampled.add(p.get(i));
      j += 1.0;
    }
  }

  resampled.redistributeBySize();

  return resampled;
}

