#ifndef INDII_ML_ODE_PARAMETERCOLLECTION_HPP
#define INDII_ML_ODE_PARAMETERCOLLECTION_HPP

#include <vector>

#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace ode {

/**
 * Collection of parameters.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 349 $
 * @date $Date: 2007-11-20 20:48:40 +0000 (Tue, 20 Nov 2007) $
 */
class ParameterCollection {
public:
  /**
   * Default constructor for restoring from serialization.
   */
  ParameterCollection();

  /**
   * Construct new parameter collection.
   *
   * @param N Number of parameters.
   */
  ParameterCollection(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~ParameterCollection();

  /**
   * Get the value of a parameter.
   *
   * @param index Index of the parameter to retrieve.
   *
   * @return The value of the parameter.
   */
  virtual double getParameter(const unsigned int index) const;

  /**
   * Set the value of a parameter.
   *
   * @param index Index of the parameter to set.
   * @param value The value to which to set the parameter.
   */
  virtual void setParameter(const unsigned int index, const double value);

private:
  /**
   * Number of parameters.
   */
  unsigned int N;

  /**
   * Parameters.
   */
  std::vector<double> ps;

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

#include <assert.h>

inline
double indii::ml::ode::ParameterCollection::getParameter(
    const unsigned int index) const {
  /* pre-condition */
  assert (index < N);

  return ps[index];
}

template<class Archive>
void indii::ml::ode::ParameterCollection::serialize(Archive& ar,
    const unsigned int version) {
  ar & N;
  ar & ps;
}

#endif

