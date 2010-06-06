#ifndef INDII_ML_FILTER_SMOOTHER_HPP
#define INDII_ML_FILTER_SMOOTHER_HPP

#include "../aux/GaussianPdf.hpp"

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Abstract smoother.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 * @param P The type of probability distribution used to represent the
 * system state.
 * 
 * @see indii::ml::filter for general usage guidelines.
 */
template <class T = unsigned int, class P = indii::ml::aux::GaussianPdf>
class Smoother {
public:
  /**
   * Constructor.
   *
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  Smoother(const T tT, const P& p_xT);

  /**
   * Destructor.
   */
  virtual ~Smoother();

  /**
   * Get the current time.
   *
   * @return \f$t_n\f$; the current time.
   */
  T getTime() const;

  /**
   * Set the current time.
   *
   * @param tn \f$t_n\f$; the current time.
   */
  void setTime(const T tn);

  /**
   * Get distribution over the state at the current time given all
   * measurements.
   *
   * @return \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_T)\big)\f$; distribution
   * over the current state given all measurements.
   */
  P& getSmoothedState();

  /**
   * Set the distribution over the state at the current time given
   * all measurements.
   *
   * @param p_xtn_ytT \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_1),\ldots,\mathbf{y}(t_T)\big)\f$; distribution
   * over the current state given all measurements.
   */
  void setSmoothedState(const P& p_xtn_ytT);

  /**
   * Apply the measurement function to the current smoothed state to
   * obtain an estimated measurement.
   *
   * @return The estimated measurement.
   */
  virtual P smoothedMeasure() = 0;
 
protected:
  /**
   * \f$t_n\f$; the current time.
   */
  T tn;

  /**
   * \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:T})\f$; smoothed density at the
   * current time.
   */
  P p_xtn_ytT;

};

    }
  }
}

template <class T, class P>
indii::ml::filter::Smoother<T,P>::Smoother(const T tT, const P& p_xT) :
    tn(tT), p_xtn_ytT(p_xT) {
  //
}

template <class T, class P>
indii::ml::filter::Smoother<T,P>::~Smoother() {
  //
}

template <class T, class P>
inline T indii::ml::filter::Smoother<T,P>::getTime() const {
  return this->tn;
}

template <class T, class P>
void indii::ml::filter::Smoother<T,P>::setTime(const T tn) {
  this->tn = tn;
}

template <class T, class P>
inline P& indii::ml::filter::Smoother<T,P>::getSmoothedState() {
  return this->p_xtn_ytT;
}

template <class T, class P>
void indii::ml::filter::Smoother<T,P>::setSmoothedState(
    const P& p_xtn_ytT) {
  this->p_xtn_ytT = p_xtn_ytT;
}

#endif

