#ifndef INDII_ML_ODE_DIFFERENTIALMODEL_HPP
#define INDII_ML_ODE_DIFFERENTIALMODEL_HPP

#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace ode {

/**
 * AdaptiveRungeKutta compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 542 $
 * @date $Date: 2008-08-31 14:44:14 +0100 (Sun, 31 Aug 2008) $
 */
class DifferentialModel {
public:
  /**
   * Default constructor for restoring from serialization.
   */
  DifferentialModel();

  /**
   * Constructor.
   *
   * @param dimensions Number of state variables in the system.
   */
  DifferentialModel(const unsigned int dimensions);

  /**
   * Destructor.
   */
  virtual ~DifferentialModel();

  /**
   * Calculate derivatives of the system at a given time.
   *
   * @param t \f$t\f$; the time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.

   * @param dydt Array into which to write the calculated derivatives
   * \f$\frac{d}{dt}y_i(t)\f$ for state variables at time \f$t\f$.
   *
   * @see indii::ml::aux::arrayToVector and
   * indii::ml::aux::vectorToArray for convenient methods for
   * converting between C/C++ arrays and indii::ml::aux::vector.
   * 
   * @see gsl_odeiv_system data type and gsl_odeiv_evolve_apply()
   * function of the @ref GSL "GSL".
   */
  virtual void calculateDerivatives(double t, const double y[],
      double dydt[]) = 0;

  /*
   * Compute Jacobian of the system of differential equations.
   *
   * @param t The time.
   * @param y The value of all state variables \f$y_i\f$ at time t.
   * @param dfdy Row-ordered array in which to write the Jacobian
   * matrix of derivatives \f$\frac{d^2y_i}{dt dy_j}\f$ for each pair
   * of state variables \f$y_i\f$ and \f$y_j\f$.
   * @param dfdt Array in which to write the calculated derivative
   * \f$\frac{d^2y_i}{dt^2}\f$ for each state variable \f$y_i\f$.
   *
   * @see gsl_odeiv_system data type and gsl_odeiv_evolve_apply()
   * function of the @ref GSL for more information.
   */
  //virtual void calculateJacobian(double t, const double y[], double *dfdy,
  //    double *dfdt) = 0;

  /**
   * Number of state variables in the system.
   *
   * @return Number of state variables in the system.
   */
  unsigned int getDimensions();

private:
  /**
   * Number of state variables in the system.
   */
  unsigned int dimensions;

  /**
   * Serialize, or restore from serialization.
   */
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version);

  /*
   * Boost.Serialization requirements.
   */
  friend class boost::serialization::access;
  
};

    }
  }
}

template<class Archive>
void indii::ml::ode::DifferentialModel::serialize(Archive& ar,
    const unsigned int version) {
  ar & dimensions;
}

#endif
