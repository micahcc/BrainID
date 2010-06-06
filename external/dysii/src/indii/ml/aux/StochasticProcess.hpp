#ifndef INDII_ML_AUX_STOCHASTICPROCESS_HPP
#define INDII_ML_AUX_STOCHASTICPROCESS_HPP

#include "vector.hpp"
#include "matrix.hpp"

#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace aux {

/**
 * Abstract stochastic process.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 540 $
 * @date $Date: 2008-08-31 14:41:46 +0100 (Sun, 31 Aug 2008) $
 */
template <class T = unsigned int>
class StochasticProcess {
public:
  /**
   * Default constructor.
   *
   * Initialises the process with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  StochasticProcess();

  /**
   * Constructor.
   *
   * @param N \f$N\f$; number of dimensions of the process.
   */
  StochasticProcess(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~StochasticProcess();

  /**
   * Get the dimensionality of the process.
   */
  unsigned int getDimensions() const;

  /**
   * Set the dimensionality of the process.
   *
   * @param N Dimensionality of the process.
   * @param preserve True to preserve the current sufficient
   * statistics of the process in the lower dimensional space,
   * false if these may be discarded.
   */
  virtual void setDimensions(const unsigned int N,
      const bool preserve = false) = 0;

  /**
   * Get the drift of the process.
   *
   * @return The drift of the process.
   */
  virtual const vector& getDrift() = 0;

  /**
   * Get the diffusion of the process.
   *
   * @return The diffusion of the process.
   */
  virtual const symmetric_matrix& getDiffusion() = 0;

  /**
   * Get the expected value of the process after a given time.
   *
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$\mathbf{\mu}(\Delta t)\f$; expected value of the process
   * after time \f$\Delta t\f$.
   */
  virtual vector getExpectation(const T delta) = 0;

  /**
   * Get the covariance of the process after a given time.
   *
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return \f$\Sigma(\Delta t)\f$; covariance of the process after
   * time \f$\Delta t\f$.
   */
  virtual symmetric_matrix getCovariance(const T delta) = 0;

  /**
   * Sample from the process.
   *
   * @param delta \f$\Delta t\f$; time step.
   *
   * @return A sample from the process after time \f$\Delta t\f$.
   */
  virtual vector sample(const T delta) = 0;

  /**
   * Calculate the density of the distribution at a given point after
   * a given time has elapsed.
   *
   * @param delta \f$\Delta t\f$; elapsed time.
   * @param x \f$\mathbf{x}\f$; the point at which to calculate the
   * density.
   *
   * @return The density of the distribution at \f$\mathbf{x}\f$ after
   * time \f$\Delta t\f$.
   */
  virtual double densityAt(const T delta, const vector& x) = 0;

protected:
  /**
   * \f$N\f$; number of dimensions.
   */
  unsigned int N;

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

template <class T>
indii::ml::aux::StochasticProcess<T>::StochasticProcess() : N(0) {
  //
}

template <class T>
indii::ml::aux::StochasticProcess<T>::StochasticProcess(
    const unsigned int N) : N(N) {
  //
}

template <class T>
indii::ml::aux::StochasticProcess<T>::~StochasticProcess() {
  //
}

template <class T>
inline unsigned int indii::ml::aux::StochasticProcess<T>::getDimensions()
    const {
  return N;
} 

template <class T>
template <class Archive>
void indii::ml::aux::StochasticProcess<T>::serialize(Archive& ar,
    const unsigned int version) {
  ar & N;
}

#endif

