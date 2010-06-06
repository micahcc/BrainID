#ifndef DOUBLEWELL_HPP
#define DOUBLEWELL_HPP

#include "indii/ml/sde/StochasticDifferentialModel.hpp"

/**
 * Double-well stochastic differential system.
 *
 * \f[
 *   dy = 4y(\theta - y^2)\,dt + \sigma\,dW
 * \f]
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 579 $
 * @date $Date: 2008-12-15 17:02:48 +0000 (Mon, 15 Dec 2008) $
 */
class DoubleWell : public indii::ml::sde::StochasticDifferentialModel<> {
public:
  /**
   * Constructor.
   */
  DoubleWell();

  /**
   * Destructor.
   */
  virtual ~DoubleWell();

  virtual indii::ml::aux::vector calculateDrift(const double ts,
      const indii::ml::aux::vector &y);

  virtual indii::ml::aux::matrix calculateDiffusion(const double ts,
      const indii::ml::aux::vector &y);

  static const double THETA = 1.0;
  static const double SIGMA = 1.0;

};

#endif
