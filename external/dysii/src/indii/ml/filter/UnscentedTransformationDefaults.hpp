#ifndef INDII_ML_FILTER_UNSCENTEDTRANSFORMATIONDEFAULTS
#define INDII_ML_FILTER_UNSCENTEDTRANSFORMATIONDEFAULTS

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Default parameter settings for UnscentedTransformation.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 274 $
 * @date $Date: 2007-07-18 13:37:55 +0100 (Wed, 18 Jul 2007) $
 *
 * These defaults are based on values given in @ref Wan2000 "Wan & van der
 * Merwe (2000)".
 */
class UnscentedTransformationDefaults {
public:
    /**
     * \f$\alpha\f$; spread of the sigma points about
     * \f$\mathbf{\bar{x}}\f$.
     */
    static const double ALPHA;

    /**
     * \f$\beta\f$; incorporates prior knowledge of the distribution of
     * \f$\mathbf{x}\f$. Default value is optimal for Gaussian distributions.
     */
    static const double BETA;

    /**
     * \f$\kappa\f$; secondary scaling parameter. Default value is usual.
     */
    static const double KAPPA;

};

    }
  }
}

#endif
