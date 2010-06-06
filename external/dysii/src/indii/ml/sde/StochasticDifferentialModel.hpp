#ifndef INDII_ML_ODE_STOCHASTICDIFFERENTIALMODEL_HPP
#define INDII_ML_ODE_STOCHASTICDIFFERENTIALMODEL_HPP

#include "../aux/vector.hpp"
#include "../aux/matrix.hpp"

#include "boost/serialization/serialization.hpp"

#include <vector>

namespace indii {
  namespace ml {
    namespace sde {
/**
 * StochasticAdaptiveRungeKutta compatible model.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 565 $
 * @date $Date: 2008-09-13 22:25:02 +0100 (Sat, 13 Sep 2008) $
 *
 * @param DT Type of the diffusion matrix.
 * @param DDT Type of the diffusion partial derivative matrices.
 *
 * The model is of the form:
 *
 * \f[
 * d\mathbf{y} = \mathbf{a}(\mathbf{y},t)\,dt + B(\mathbf{y},t)\,d\mathbf{W},
 * \f]
 *
 * where \f$\mathbf{a}(\mathbf{y},t)\f$ is the drift term and
 * \f$B(\mathbf{y},t)\f$ the diffusion term. At any time \f$t\f$, the
 * time derivatives may be calculated as:
 *
 * \f[
 *   \frac{dy_k}{dt} = a_k(\mathbf{y},t) -
 *   \frac{1}{2}\sum_{ij}B_{ij}(\mathbf{y},t)\frac{\partial
 *   B_{kj}(\mathbf{y},t)}{\partial y_i} + \frac{1}{\Delta
 *   t}\sum_{i}B_{ki}(\mathbf{y},t)\Delta W_i,
 * \f]
 *
 * or in matrix form:
 *
 * \f[
 *   \frac{d\mathbf{y}}{dt} = \mathbf{a}(\mathbf{y},t) -
 *   \frac{1}{2}\sum_i \frac{\partial B(\mathbf{y},t)}{\partial
 *   y_i}B_{i,*}(\mathbf{y},t)^T + \frac{1}{\Delta t}
 *   B(\mathbf{y},t)\Delta \mathbf{W},
 * \f]
 *
 * where \f$\Delta t\f$ is the time step, \f$\Delta \mathbf{W}\f$ the
 * Wiener increment and \f$B_{i,*}(\mathbf{y},t)\f$ the \f$i\f$th row of
 * \f$B(\mathbf{y},t)\f$.
 *
 * This is all calculated by StochasticAdaptiveRungeKutta, although
 * the model must provide \f$\mathbf{a}(\mathbf{y},t)\f$,
 * \f$B(\mathbf{y},t)\f$ and optionally \f$\frac{\partial
 * B_{kj}(\mathbf{y},t)}{\partial y_i}\f$ through implementations of
 * the calculateDrift(), calculateDiffusion() and
 * calculateDiffusionDerivatives() functions, respectively.
 */
template <class DT = indii::ml::aux::matrix,
    class DDT = indii::ml::aux::zero_matrix>
class StochasticDifferentialModel {
public:
  /**
   * Default constructor for restoring from serialization.
   */
  StochasticDifferentialModel();

  /**
   * Constructor.
   *
   * @param dimensions Number of state variables in the system.
   *
   * The Wiener process noise is assumed to have the same
   * dimensionality as the state.
   */
  StochasticDifferentialModel(const unsigned int dimensions);

  /**
   * Constructor.
   *
   * @param dimensions Number of state variables in the system.
   * @param noiseDimensions Number of noise variables in the system.
   */
  StochasticDifferentialModel(const unsigned int dimensions,
      const unsigned int noiseDimensions);

  /**
   * Destructor.
   */
  virtual ~StochasticDifferentialModel();

  /**
   * Number of state variables in the system.
   *
   * @return the number of state variables in the system.
   */
  unsigned int getDimensions();

  /**
   * Number of noise variables in the system.
   *
   * @return the number of noise variables in the system.
   */
  unsigned int getNoiseDimensions();

  /**
   * Calculate drift term of the system at a given time.
   *
   * @param ts \f$t_s\f$; the proposed new time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.
   * @param a \f$\mathbf{a}(\mathbf{y}, t)\f$; vector into which to write
   * the drift term at time \f$t\f$. This is uninitialised.
   *
   * Default implementation calls the older, now deprecated method
   * calculateDrift(const double, const indii::ml::aux::vector&).
   */
  virtual void calculateDrift(const double ts,
      const indii::ml::aux::vector& y, indii::ml::aux::vector& a);

  /**
   * Calculate diffusion term of the system at a given time.
   *
   * @param ts \f$t_s\f$; the proposed new time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.
   * @param B \f$B(\mathbf{y}, t)\f$; matrix into which to write the
   * diffusion term at time \f$t\f$. This is either cleared or contains
   * the last diffusion calculation when called.
   *
   * Default implementation calls the older, now deprecated method
   * calculateDiffusion(const double, const indii::ml::aux::vector&).
   */
  virtual void calculateDiffusion(const double ts,
      const indii::ml::aux::vector& y, DT& B);

  /**
   * Calculate the partial derivatives of the diffusion term with
   * respect to each state variable at a given time.
   *
   * @param ts \f$t_s\f$; the proposed new time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.
   * @param dBdy A vector of \f$N\f$ matrices into which to write the
   * result, where matrix \f$i\f$ is
   * \f$\frac{\partial B(\mathbf{y},t)}{\partial y_i}\f$ at time
   * \f$t\f$. Each matrix is either cleared or contains the last
   * partial derivative calculation when called.
   */
  virtual void calculateDiffusionDerivatives(const double ts,
      const indii::ml::aux::vector& y, std::vector<DDT>& dBdy);

  /**
   * Calculate drift term of the system at a given time.
   *
   * @param ts \f$t_s\f$; the proposed new time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.
   *
   * @return \f$\mathbf{a}(\mathbf{y}, t)\f$; the drift term at time
   * \f$t\f$.
   *
   * @deprecated Use calculateDrift(const double,
   * const indii::ml::aux::vector&, indii::ml::aux::vector&)
   */
  virtual indii::ml::aux::vector calculateDrift(const double ts,
      const indii::ml::aux::vector& y);

  /**
   * Calculate diffusion term of the system at a given time.
   *
   * @param ts \f$t_s\f$; the proposed new time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.
   *
   * @return \f$B(\mathbf{y}, t)\f$; the diffusion term at time
   * \f$t\f$.
   *
   * @deprecated Use calculateDiffusion(const double,
   * const indii::ml::aux::vector&, DT&)
   */
  virtual DT calculateDiffusion(const double ts,
      const indii::ml::aux::vector& y);

  /**
   * Calculate the partial derivatives of the diffusion term with
   * respect to each state variable at a given time.
   *
   * @param ts \f$t_s\f$; the proposed new time.
   * @param y \f$\mathbf{y}(t)\f$; the values of all state variables
   * at time \f$t\f$.
   *
   * @return A vector of \f$N\f$ matrices, where matrix \f$i\f$ is
   * \f$\frac{\partial B(\mathbf{y},t)}{\partial y_i}\f$ at time
   * \f$t\f$. In the special case that \f$B(\mathbf{y}, t)\f$ is not
   * dependent on \f$\mathbf{y}\f$, may return an empty
   * std::vector<aux::matrix>. This is the behaviour of the default
   * implementation and has the same effect as returning a vector of
   * \f$N\f$ zero matrices.
   *
   * @deprecated Use calculateDiffusionDerivatives(const double,
   * const indii::ml::aux::vector&, std::vector<DDT>)
   */
  virtual std::vector<DDT> calculateDiffusionDerivatives(
      const double ts, const indii::ml::aux::vector& y);

private:
  /**
   * Number of state variables in the system.
   */
  unsigned int dimensions;

  /**
   * Number of noise variables in the system.
   */
  unsigned int noiseDimensions;

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

template <class DT, class DDT>
indii::ml::sde::StochasticDifferentialModel<DT,DDT>::StochasticDifferentialModel() {
  this->dimensions = 0;
  this->noiseDimensions = 0;
}

template <class DT, class DDT>
indii::ml::sde::StochasticDifferentialModel<DT,DDT>::StochasticDifferentialModel(
    const unsigned int dimensions) {
  this->dimensions = dimensions;
  this->noiseDimensions = dimensions;
}

template <class DT, class DDT>
indii::ml::sde::StochasticDifferentialModel<DT,DDT>::StochasticDifferentialModel(
    const unsigned int dimensions, const unsigned int noiseDimensions) {
  this->dimensions = dimensions;
  this->noiseDimensions = noiseDimensions;
}

template <class DT, class DDT>
indii::ml::sde::StochasticDifferentialModel<DT,DDT>::~StochasticDifferentialModel() {
  //
}

template <class DT, class DDT>
inline unsigned int
    indii::ml::sde::StochasticDifferentialModel<DT,DDT>::getDimensions() {
  return dimensions;
}

template <class DT, class DDT>
inline unsigned int indii::ml::sde::StochasticDifferentialModel<DT,DDT>::getNoiseDimensions() {
  return noiseDimensions;
}

template <class DT, class DDT>
void indii::ml::sde::StochasticDifferentialModel<DT,DDT>::calculateDrift(
    const double ts, const indii::ml::aux::vector& y,
    indii::ml::aux::vector& a) {
  noalias(a) = calculateDrift(ts, y);
}

template <class DT, class DDT>
void indii::ml::sde::StochasticDifferentialModel<DT,DDT>::calculateDiffusion(
    const double ts, const indii::ml::aux::vector& y, DT& B) {
  noalias(B) = calculateDiffusion(ts, y);
}

template <class DT, class DDT>
void indii::ml::sde::StochasticDifferentialModel<DT,DDT>::calculateDiffusionDerivatives(
    const double ts, const indii::ml::aux::vector& y,
    std::vector<DDT>& dBdy) {
  dBdy = calculateDiffusionDerivatives(ts, y);
}

template <class DT, class DDT>
indii::ml::aux::vector
    indii::ml::sde::StochasticDifferentialModel<DT,DDT>::calculateDrift(
    const double ts, const indii::ml::aux::vector& y) {
  indii::ml::aux::vector a(dimensions);
  a.clear();
  
  return a;
}

template <class DT, class DDT>
DT indii::ml::sde::StochasticDifferentialModel<DT,DDT>::calculateDiffusion(
    const double ts, const indii::ml::aux::vector& y) {
  DT B(dimensions, noiseDimensions);
  B.clear();
  
  return B;
}

template <class DT, class DDT>
std::vector<DDT> indii::ml::sde::StochasticDifferentialModel<DT,DDT>::calculateDiffusionDerivatives(
    const double ts, const indii::ml::aux::vector& y) {
  std::vector<DDT> nil;

  return nil;
}

template <class DT, class DDT>
template<class Archive>
void indii::ml::sde::StochasticDifferentialModel<DT,DDT>::serialize(Archive& ar,
    const unsigned int version) {
  ar & dimensions;
  ar & noiseDimensions;
}

#endif

