//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "Random.hpp"
//#endif

#include "boost/mpi.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

#include <time.h>
#include <assert.h>

using namespace indii::ml::aux;

namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

bool Random::isInit = false;

gsl_rng* Random::rng = NULL;

void Random::seed(unsigned int seed) {
  if (!isInit) {
    init();
  }
  gsl_rng_set(rng, seed);
}

void Random::init() {
  int seed;

  /* construct random number generator */
  gsl_rng_env_setup();
  //rng = gsl_rng_alloc(gsl_rng_ranlxd2); // best randomness
  rng = gsl_rng_alloc(gsl_rng_mt19937); // best randomness/speed tradeoff

  /* select seed */
  seed = time(NULL);
  if (boost::mpi::environment::initialized()) {
    /* ensure two nodes aren't seeded with the same number */
    boost::mpi::communicator world;
    const int rank = world.rank();
    seed += 1000 * rank;
  }

  /* seed random number generator */
  gsl_rng_set(rng, seed);
  isInit = true;

  /* post-condition */
  assert (isInit);
}

void Random::terminate() {
  if (isInit) {
    gsl_rng_free(rng);
  }
}

vector Random::unitVector(const unsigned int N) {
  if (!isInit) {
    init();
  }
  
  aux::vector s(N);
  double x[N]; 
  gsl_ran_dir_nd(rng, N, x);
  arrayToVector(x, s);
  
  return s;
}

matrix Random::orthonormalMatrix(const unsigned int N) {
  if (!isInit) {
    init();
  }
  
  identity_matrix I(N,N);
  matrix random(N,N);  // random matrix
  matrix Q(I), H(I);
  vector v(N), tau(N);
  unsigned int i, j;
  int ierr;

  /* create random matrix */
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i++) {
      random(i,j) = Random::uniform(-1.0, 1.0);
    }
  }
  /**
   * @todo Should check random for full rank, just in case, although it's
   * highly unlikely that the set is linearly dependent.
   */

  /* QR factorisation */
  ierr = lapack::geqrf(random, tau);
  assert (ierr == 0);

  /* rebuild Q from reflectors (see LAPACK documentation for how Q is
     returned) */
  ublas::triangular_adaptor<aux::matrix, ublas::lower> L(random);

  for (i = 0; i < N; i++) {
    v = column(L,i);
    v(i) = 1.0;
 
    H = I - tau(i) * ublas::outer_prod(v,v);
    Q = prod(Q,H);
  }

  return Q;
}

