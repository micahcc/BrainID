//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "StratifiedParticleResampler.hpp"

namespace aux = indii::ml::aux;

using namespace indii::ml::filter;


#include <iostream>
StratifiedParticleResampler::StratifiedParticleResampler(
    const unsigned int P) : P(P), method(CUSTOM) 
{
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    rng = gsl_rng_alloc(gsl_rng_ranlxd2);
    {
        unsigned int seed;
        FILE* file = fopen("/dev/urandom", "r");
        fread(&seed, 1, sizeof(unsigned int), file);
        fclose(file);
        gsl_rng_set(rng, seed^rank);
        std::cout << "Stratified\n";
        std::cout << "Seeding with " << (unsigned int)(seed^rank) << "\n";
    }
}

StratifiedParticleResampler::~StratifiedParticleResampler() {
  //
}

void StratifiedParticleResampler::setNumParticles(const unsigned int P) {
  this->P = P;
}

aux::DiracMixturePdf StratifiedParticleResampler::resample(
    aux::DiracMixturePdf& p) 
{
    aux::DiracMixturePdf resampled(p.getDimensions());
    switch(method) {
        case DETERMINISTIC:
            resample_deterministic(p, resampled);
        case MIXTURE:
            resample_mixture(p, resampled);
        case CUSTOM1:
            resample_custom1(p, resampled);
        case CUSTOM2:
            resample_custom2(p, resampled);
    }
    return resampled;
}

/** 
 * Finds the lowest index for which term > arr[index] 
 * 
 * @param arr - aray to search, should be increaseing
 * @param term - level to search for
 * @return index for which arr[index] >= term >= arr[index-1]
*/
int search(const std::vector<double>& arr, double term)
{
    std::vector<double>::iterator pos = lower_bound(arr.begin(), arr.end(), term);
    int index = pos - arr.begin();

    //deal with 0 weighted points (in which case arr[i-1]=arr[i], but 
    //  i has zero weight
//    while(index != 0 && arr[index] == arr[index-1])
//        index--;

    return index;
}

void StratifiedParticleResampler::resample_custom(
            const aux::DiracMixturePdf& p, aux::DiracMixturePdf& resampled) 
{
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

  unsigned int P = this->P;
  if (P == 0) {
    P = p.getDistributedSize();
  }

  /* Prep old distribution */
  p.gatherToNode(0);
  const std::vector<double>& Ws = p.getCumulativeWeights();

  /* Create Containers for new distribution */
  aux::vector ws_r(P, 1./P);
  std::vector<aux::vector>& xs_r = resampled.getAll();
  xs_r.resize(P, false);
  
  int index = 0;
  for(unsigned int i = 0 ; i < P ; i++) {
    tmp = gsl_ran_uniform(rng, 0, Ws.back());
    index = search(Ws, tmp);

    xs_r[i] = p.get(index);
    assert(p.getWeight(index) != 0);
  }

  //components are already set, xs_r is a reference to it
  resampled.setWeights(ws_r);
  resampled.redistributeBySize();
}

void StratifiedParticleResampler::resample_mixture(
            const aux::DiracMixturePdf& p, aux::DiracMixturePdf& resampled) 
{
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

  unsigned int P = this->P;
  if (P == 0) {
    P = p.getDistributedSize();
  }

  /* Create Containers for new distribution */
  aux::vector ws_r(P, 1./P);
  std::vector<aux::vector>& xs_r = resampled.getAll();
  xs_r = p.distributedSample(P);
  
  //components are already set, xs_r is a reference to it
  resampled.setWeights(ws_r);
  resampled.redistributeBySize();
}

void StratifiedParticleResampler::resample_deterministic(
            const aux::DiracMixturePdf& p, aux::DiracMixturePdf& resampled) {
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

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
}


