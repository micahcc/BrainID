#ifndef INDII_ML_ODE_FUNCTIONCOLLECTION_HPP
#define INDII_ML_ODE_FUNCTIONCOLLECTION_HPP

#include "FunctionModel.hpp"
#include "FunctionStatic.hpp"

#include "boost/serialization/split_member.hpp"

#include <vector>

namespace indii {
  namespace ml {
    namespace ode {

/**
 * Collection of functions.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 355 $
 * @date $Date: 2007-11-26 00:19:09 +0000 (Mon, 26 Nov 2007) $
 */
class FunctionCollection {
public:
  /**
   * Default constructor for restoration from serialization.
   */
  FunctionCollection();

  /**
   * Construct new collection of functions.
   *
   * @param N Number of functions.
   */
  FunctionCollection(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~FunctionCollection();

  /**
   * Get the value of a function.
   *
   * @param index Index of the function.
   *
   * @return The value of the function the last time it was evaluated
   * by a call to evaluate().
   */
  virtual double getFunction(const unsigned int index) const;

  /**
   * Set a function in the collection.
   *
   * @param index Index of the function to set.
   * @param value The function.
   */
  virtual void setFunction(const unsigned int index, f_t value);

  /**
   * Set a function in the collection using an object rather than a
   * static function.
   *
   * @param index Index of the function to set.
   * @param value Object encapsulating the function. The caller
   * retains ownership.
   */
  virtual void setFunction(const unsigned int index, FunctionModel* value);

  /**
   * Is a function static?
   *
   * @param index Index of the function to check.
   *
   * @return True if the given function is static, false otherwise.
   */
  bool isStatic(const unsigned int index);

  /**
   * Set the object passed as an argument to all static functions when
   * called.
   *
   * @param object The object. The caller retains ownership.
   */
  virtual void setObject(void* object);

  /**
   * Evaluate all functions in the collection for a given time and
   * state, and store the results for subsequent calls to
   * getFunction().
   *
   * @param t The time.
   * @param y Array holding values of state variables at time t.
   */
  void evaluateAll(const double t, const double y[]);

private:
  /**
   * Number of functions.
   */
  unsigned int N;

  /**
   * Static functions.
   */
  std::vector<f_t*> fs;

  /**
   * Function models.
   */
  std::vector<FunctionModel*> fms;

  /**
   * Function cached values.
   */
  std::vector<double> fvs;

  /**
   * Record of static functions.
   */
  std::vector<bool> statics;

  /**
   * Object passed as argument to all calls to static functions.
   */
  void* object;

  /**
   * Serialize. Note that serialization of a FunctionCollection object
   * does not include static functions stored in it, only the
   * structure in which the functions are stored. This is due to
   * difficulties in serializing function pointers. After restoration,
   * the functions should be restored with calls to setFunction() and
   * setObject().
   */
  template<class Archive>
  void save(Archive& ar, const unsigned int version) const;

  /**
   * Restore from serialization.
   */
  template<class Archive>
  void load(Archive& ar, const unsigned int version);

  /*
   * Boost.Serialization requirements.
   */
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  friend class boost::serialization::access;

};

    }
  }
}

#include "boost/serialization/vector.hpp"

#include <assert.h>

inline
double indii::ml::ode::FunctionCollection::getFunction(
    const unsigned int index) const {
  /* pre-condition */
  assert (index < N);

  return fvs[index];
}

template<class Archive>
void indii::ml::ode::FunctionCollection::save(Archive& ar,
    const unsigned int version) const {
  unsigned int i;

  ar & N;
  ar & fvs;
  ar & fms;

  /**
   * @todo Segfault when serializing statics directly, so do this way...
   */
  for (i = 0; i < N; i++) {
    const bool b = statics[i];
    ar & b;
  }
}

template<class Archive>
void indii::ml::ode::FunctionCollection::load(Archive& ar,
    const unsigned int version) {
  unsigned int i;
  bool b;

  ar & N;
  ar & fvs;
  ar & fms;

  fs.resize(N);
  statics.resize(N);

  for (i = 0; i < N; i++) {
    ar & b;
    statics[i] = b;
  }

  for (i = 0; i < N; i++) {
    fs[i] = NULL;
    if (statics[i]) {
      fms[i] = NULL;
    }
  }

  object = this;
}

#endif
