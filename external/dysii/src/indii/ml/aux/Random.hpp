#ifndef INDII_ML_AUX_RANDOM_HPP
#define INDII_ML_AUX_RANDOM_HPP

#include "vector.hpp"
#include "matrix.hpp"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h>

namespace indii {
  namespace ml {
    namespace aux {
/**
 * %Random numbers.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 471 $
 * @date $Date: 2008-05-30 14:21:19 +0100 (Fri, 30 May 2008) $
 *
 * This class makes use of the random number generation features of
 * the @ref GSL "GSL", in particular making use of the MT19937
 * generator of @ref Matsumoto1998 "Matsumoto & Nishimura (1998)". If
 * not explicitly seeded, it is seeded with the current time as
 * returned by the system's time() function the first time one of the
 * provided functions is called.
 *
 * More information on the implementation of this random number
 * generator is available in the GSL manual.
 *
 * @section Random_references References
 *
 * @anchor GSL
 * The GNU Scientific Library (GSL). http://www.gnu.org/software/gsl/
 * 
 * @anchor Matsumoto1998 Matsumoto, M. and Nishimura,
 * T. Mersenne Twister: A 623-dimensionally equidistributed
 * uniform pseudorandom number generator. <i>ACM Transactions on
 * Modeling and Computer Simulation</i>, <b>1998</b>, 8, 3-30.
 */
class Random {
public:
  /**
   * Seed the random number generator.
   *
   * @param seed Seed value.
   *
   * Seeds the random number generator for future use using the given
   * seed value. If not explicitly seeded, the generator is seeded
   * with the current time the first time it is used.
   */
  static void seed(unsigned int seed);

  /**
   * Generate a random number from a uniform distribution over the
   * given interval.
   *
   * @param lower Lower bound on the interval.
   * @param upper Upper bound on the interval.
   *
   * @return The random number.
   */
  static double uniform(const double lower = 0.0, const double upper = 1.0);

  /**
   * Generate a random number from a Gaussian distribution with the
   * given mean and standard deviation.
   *
   * @param mean Mean of the distribution.
   * @param sigma Standard deviation of the distribution.
   *
   * @return The random number.
   */
  static double gaussian(const double mean = 0.0, const double sigma = 1.0);

  /**
   * Generate a boolean value from a Bernoulli distribution.
   *
   * @param p Probability of true, between 0 and 1 inclusive.
   *
   * @return The random boolean value, 1 for true, 0 for false.
   */
  static unsigned int bernoulli(const double p = 0.5);

  /**
   * Generate a random unit vector.
   *
   * @param N Size of unit vector.
   *
   * @return Random unit vector in @p N dimensions.
   */
  static vector unitVector(const unsigned int N);

  /**
   * Generate a random orthonormal matrix.
   *
   * @param N Size of matrix.
   *
   * @return Random @p N by @p N orthonormal matrix.
   */
  static matrix orthonormalMatrix(const unsigned int N);

private:
  /**
   * Is random number generator initialised?
   */
  static bool isInit;

  /**
   * Random number generator.
   */
  static gsl_rng* rng;

  /**
   * Initialise random number generator.
   */
  static void init();

  /**
   * Terminate random number generator.
   */
  static void terminate();

};

    }
  }
}

inline double indii::ml::aux::Random::uniform(const double lower,
    const double upper) {
  if (!isInit) {
    init();
  }
  return gsl_ran_flat(rng, lower, upper);
}

inline double indii::ml::aux::Random::gaussian(const double mean,
    const double sigma) {
  if (!isInit) {
    init();
  }
  return (sigma == 0.0) ? mean : mean + gsl_ran_gaussian(rng, sigma);
}

inline unsigned int indii::ml::aux::Random::bernoulli(const double p) {
  /* pre-condition */
  assert (p >= 0.0 && p <= 1.0);

  if (!isInit) {
    init();
  }  
  return gsl_ran_bernoulli(rng, p);
}

#endif

