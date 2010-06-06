#ifndef INDII_ML_FILTER_TWOFILTERSMOOTHER_HPP
#define INDII_ML_FILTER_TWOFILTERSMOOTHER_HPP

#include "../aux/vector.hpp"
#include "../aux/Pdf.hpp"

#include "Smoother.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Abstract smoother for estimating the state of a system by fusing
 * forward and backward filtering passes.
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
class TwoFilterSmoother : public Smoother<T,P> {
public:
  /**
   * Constructor.
   *
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  TwoFilterSmoother(const T tT, const P& p_xT);

  /**
   * Destructor.
   */
  virtual ~TwoFilterSmoother();

  /**
   * Get distribution over the state at the current time given present
   * and future measurements.
   *
   * @return \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_n),\ldots,\mathbf{y}(t_T)\big)\f$; distribution
   * over the current state given present and future measurements.
   */
  P& getBackwardFilteredState();

  /**
   * Set distribution over the state at the current time given present
   * and future measurements.
   *
   * @param p_xtn_ytn \f$P\big(\mathbf{x}(t_n)\, |
   * \,\mathbf{y}(t_n),\ldots,\mathbf{y}(t_T)\big)\f$; distribution
   * over the current state given present and future measurements.
   */
  void setBackwardFilteredState(const P& p_xtn_ytn);

  /**
   * Rewind system to time of previous measurement and
   * smooth. Performs the backward filtering step and fuses this with
   * the given prediction from the forward filtering step to produce
   * the smoothed prediction.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the
   * system. This must be less than the current time \f$t_{n+1}\f$.
   * @param ytn \f$\mathbf{y}_n\f$; measurement at time \f$t_n\f$.
   * @param p_xtn_ytn \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n})\f$;
   * forward filter density.
   */
  virtual void smooth(const T tn, const aux::vector& ytn,
      const P& p_xtn_ytn) = 0;

  /**
   * Apply the measurement function to the current filtered state to
   * obtain an estimated measurement.
   *
   * @return The estimated measurement.
   */
  virtual aux::GaussianPdf backwardMeasure() = 0;

protected:
  /**
   * \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{n:T})\f$; backward filter density.
   */
  P p_xtn_ytn_b;

};

    }
  }
}

template <class T, class P>
indii::ml::filter::TwoFilterSmoother<T,P>::TwoFilterSmoother(const T tT,
    const P& p_xT) : Smoother<T,P>(tT, p_xT), p_xtn_ytn_b(p_xT) {
  //
}

template <class T, class P>
indii::ml::filter::TwoFilterSmoother<T,P>::~TwoFilterSmoother() {
  //
}

template <class T, class P>
inline P& indii::ml::filter::TwoFilterSmoother<T,P>::getBackwardFilteredState() {
  return this->p_xtn_ytn_b;
}

template <class T, class P>
void indii::ml::filter::TwoFilterSmoother<T,P>::setBackwardFilteredState(
    const P& p_xtn_ytn) {
  this->p_xtn_ytn_b = p_xtn_ytn;
}

#endif

