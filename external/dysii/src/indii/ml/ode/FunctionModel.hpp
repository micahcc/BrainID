#ifndef INDII_ML_ODE_FUNCTIONMODEL_HPP
#define INDII_ML_ODE_FUNCTIONMODEL_HPP

#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace ode {

/**
 * Function specification. This class encapsulates a time-dependent
 * function. Instantiating an object of a class derived from this is
 * one way to supply a function to a FunctionCollection object. The
 * other is to define a static function of type f_t().
 */
class FunctionModel {
public:
  /**
   * Destructor.
   */
  virtual ~FunctionModel();

  /**
   * Evaluate the function.
   *
   * @param t The time.
   * @param y State variable values at time t.
   *
   * @return The calculated value of the function given the arguments.
   */
  virtual double evaluate(const double t, const double y[]) = 0;

private:
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
void indii::ml::ode::FunctionModel::serialize(Archive& ar,
    const unsigned int version) {
  //
}

#endif
