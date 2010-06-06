#ifndef INDII_ML_ODE_FUNCTIONSTATIC_HPP
#define INDII_ML_ODE_FUNCTIONSTATIC_HPP

namespace indii {
  namespace ml {
    namespace ode {

/**
 * Function specification. This specifies the static signature of a
 * time-dependent function. It is one way in which a function may be
 * supplied to a FunctionCollection object, the other being by
 * instantiating an object of type FunctionModel.
 *
 * @param t The time.
 * @param y State variable values at time t.
 * @param object Object passed to the function via the calling
 * object. This is usually the calling object itself, allowing
 * convenient access to related parameters and functions that may be
 * required during evaluation.
 *
 * @return The calculated value of the function given the arguments.
 */
typedef double f_t(double t, const double y[], void* object);

    }
  }
}

#endif
