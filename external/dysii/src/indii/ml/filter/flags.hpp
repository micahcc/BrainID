#ifndef INDII_ML_FILTER_FLAGS_HPP
#define INDII_ML_FILTER_FLAGS_HPP

/**
 * @file flags.hpp
 *
 * Optimisation flags for KernelForwardBackwardSmoother and
 * KernelTwoFilterSmoother.
 */

namespace indii {
  namespace ml {
    namespace filter {
  /**
   * Optimisation flags. These may be ORed and passed to the smooth()
   * methods to trigger optimisations relevant in specific circumstances.
   */
  enum Flags {
    /**
     * Do not standardise kernel density evaluations.
     *
     * The support of \f$p(\mathbf{x}_{n+1}\,|\,\mathbf{y}_{1:n})\f$ and
     * \f$p(\mathbf{x}_{n+1}\,|\,\mathbf{y}_{1:T})\f$ is assumed equivalent.
     */
    NO_STANDARDISATION = 1,
    
    /**
     * Assume no resampling was performed by the %filter between times
     * \f$t_n\f$ and \f$t_{n+1}\f$.
     *
     * In the case that the %filter density is used as proposal distribution,
     * this allows reuse of the components supporting \f$p(\mathbf{x}_{n+1}\,
     * |\,\mathbf{y}_{1:n})\f$ as the propagations of the proposal particles
     * from \f$q(\mathbf{x}_n) = p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n})\f$.
     * In addition, if NO_STANDARDISATION is set, a self-tree kernel density
     * evaluation is performed.
     */
    NO_RESAMPLING = 2,
    
    /**
     * Assume that the proposal distribution is the same as the last call
     * to smooth().
     *
     * In this case, samples from the proposal distribution are reused from
     * the last call.
     */
    SAME_PROPOSAL = 4,
    
    /**
     * If SAME_PROPOSAL is set, additionally reuse the propagations of the
     * proposal samples from the last call.
     */
    SAME_PROPAGATIONS = 8,

    /**
     * Use filter-smoother when applicable. This is significantly faster,
     * but may reduce sampling effectiveness.
     */
    FILTER_SMOOTHER = 16
    
  };

    }
  }
}

#endif

