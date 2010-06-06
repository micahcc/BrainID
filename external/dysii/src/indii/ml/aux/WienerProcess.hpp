#ifndef INDII_ML_AUX_WIENERPROCESS_HPP
#define INDII_ML_AUX_WIENERPROCESS_HPP

#include "StochasticProcess.hpp"

namespace indii {
  namespace ml {
    namespace aux {

/**
 * Multivariate Wiener process.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 565 $
 * @date $Date: 2008-09-13 22:25:02 +0100 (Sat, 13 Sep 2008) $
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 */
template <class T = unsigned int>
class WienerProcess : public StochasticProcess<T> {
public:
  /**
   * Default constructor.
   *
   * Initialises the Wiener process with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  WienerProcess();

  /**
   * Construct Wiener process of given dimensionality.
   *
   * @param N \f$N\f$; number of dimensions in the process.
   *
   * The drift is initialised to zero and the diffusion to the
   * identity matrix, giving the standard Wiener process.
   */
  WienerProcess(unsigned int N);

  /**
   * Destructor.
   */
  virtual ~WienerProcess();

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  /**
   * Get the drift of the process.
   *
   * @return The drift of the process.
   */
  virtual const vector& getDrift() const;

  /**
   * Get the diffusion of the process.
   *
   * @return The diffusion of the process.
   */
  virtual const symmetric_matrix& getDiffusion() const;

  virtual const vector& getDrift();

  virtual const symmetric_matrix& getDiffusion();

  virtual vector getExpectation(const T delta);

  virtual symmetric_matrix getCovariance(const T delta);

  /**
   * Get the expected value of the process after a given time has
   * elapsed.
   *
   * @param delta \f$\Delta t\f$; elapsed time.
   *
   * @return \f$\mathbf{\mu}(\Delta t)\f$; expected value of the process
   * after time \f$\Delta t\f$.
   */
  virtual vector getExpectation(const T delta) const;

  /**
   * Get the covariance of the process after a given time has elapsed.
   *
   * @param delta \f$\Delta t\f$; elapsed time.
   *
   * @return \f$\Sigma(\Delta t)\f$; covariance of the process after
   * time \f$\Delta t\f$.
   */
  virtual symmetric_matrix getCovariance(const T delta) const;

  /**
   * Sample from the process after a given time has elapsed.
   *
   * @param delta \f$\Delta t\f$; elapsed time.
   *
   * This is performed using the following process:
   *
   * @li Let \f$\mathbf{z}\f$ be a vector of \f$N\f$ independent
   * normal variates with mean zero and variance \f$\Delta t\f$.
   *
   * @li Then the sample is \f$\mathbf{x} = \Delta t \mathbf{\mu} +
   * L\mathbf{z}\f$, where \f$L\f$ is the Cholesky decomposition of
   * the diffusion.
   *
   * @return The sample \f$\mathbf{x}\f$
   */
  virtual vector sample(const T delta);

  /**
   * Calculate the density of the distribution at a given point after
   * a given time has elapsed:
   *
   * \f[p(\mathbf{x},\Delta t) =
   *   \frac{1}{\sqrt{(2\pi\Delta t)^N|\Sigma|}}
   *   \exp\big({-\frac{1}{2\Delta t}(\mathbf{x}-\mathbf{\mu})^T \Sigma^{-1} 
   *   (\mathbf{x}-\mathbf{\mu})\big)}
   * \f]
   *
   * @param delta \f$\Delta t\f$; elapsed time.
   * @param x \f$\mathbf{x}\f$; the point at which to calculate the
   * density.
   *
   * @return The density of the distribution at \f$\mathbf{x}\f$ after
   * time \f$\Delta t\f$.
   */
  virtual double densityAt(const T delta, const vector& x);

private:
  /**
   * Drift.
   */
  vector mu;
  
  /**
   * Diffusion.
   */
  symmetric_matrix sigma;

  /**
   * Serialize.
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

#include "Random.hpp"

#include "boost/serialization/base_object.hpp"

#include <assert.h>

using namespace indii::ml::aux;

namespace ublas = boost::numeric::ublas;

template <class T>
WienerProcess<T>::WienerProcess() : mu(0), sigma(0) {
  //
}

template <class T>
WienerProcess<T>::WienerProcess(unsigned int N) : StochasticProcess<T>(N),
    mu(N), sigma(N) {
  noalias(mu) = zero_vector(N);
  noalias(sigma) = identity_matrix(N,N);
}

template <class T>
WienerProcess<T>::~WienerProcess() {
  //
}

template <class T>
void WienerProcess<T>::setDimensions(const unsigned int N,
    const bool preserve) {
  this->N = N;
  mu.resize(N, true); // force preservation, or object will be invalid
  sigma.resize(N, true);
}

template <class T>
vector WienerProcess<T>::getExpectation(const T delta) const {
  return mu;
}  

template <class T>
symmetric_matrix WienerProcess<T>::getCovariance(const T delta) const {
  return delta * sigma;
}

template <class T>
vector WienerProcess<T>::getExpectation(const T delta) {
  return mu;
}  

template <class T>
symmetric_matrix WienerProcess<T>::getCovariance(const T delta) {
  return delta * sigma;
}

template <class T>
const vector& WienerProcess<T>::getDrift() {
  return mu;
}  

template <class T>
const symmetric_matrix& WienerProcess<T>::getDiffusion() {
  return sigma;
}

template <class T>
const vector& WienerProcess<T>::getDrift() const {
  return mu;
}  

template <class T>
const symmetric_matrix& WienerProcess<T>::getDiffusion() const {
  return sigma;
}

template <class T>
vector WienerProcess<T>::sample(const T delta) {
  vector z(this->N);
  unsigned int i;

  for (i = 0; i < this->N; i++) {
    z(i) = Random::gaussian(0.0, sqrt(delta));
  }

  return z;
}

template <class T>
double WienerProcess<T>::densityAt(const T delta, const vector& x) {
  double deltaI = 1.0 / delta;
  double exponent = inner_prod(x,x);
  double z = sqrt(std::pow(2.0*M_PI*deltaI, static_cast<int>(this->N)));
  
  return z * exp(-0.5 * deltaI * exponent);
}

template <class T>
template <class Archive>
void indii::ml::aux::WienerProcess<T>::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<StochasticProcess<T> >(*this);
  ar & mu;
  
  /* serialization of symmetric_matrix not yet supported in ublas */
  const aux::matrix tmp(sigma);
  ar & tmp;
}

template <class T>
template <class Archive>
void indii::ml::aux::WienerProcess<T>::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<StochasticProcess<T> >(*this);
  ar & mu;
  
  /* serialization of symmetric_matrix not yet supported in ublas */
  aux::matrix tmp(this->N,this->N);
  ar & tmp;
  noalias(sigma) = boost::numeric::ublas::symmetric_adaptor<aux::matrix>(tmp);
}

#endif

