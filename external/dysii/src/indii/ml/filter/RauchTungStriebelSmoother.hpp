#ifndef INDII_ML_FILTER_RAUCHTUNGSTRIEBELSMOOTHER_HPP
#define INDII_ML_FILTER_RAUCHTUNGSTRIEBELSMOOTHER_HPP

#include "Smoother.hpp"
#include "RauchTungStriebelSmootherModel.hpp"

#include <stack>

namespace indii {
  namespace ml {
    namespace filter {

/**
 * Rauch-Tung-Striebel (RTS) smoother.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 544 $
 * @date $Date: 2008-09-01 15:04:39 +0100 (Mon, 01 Sep 2008) $
 *
 * @param T The type of time.
 *
 * RauchTungStriebelSmoother is suitable for models with linear
 * transition and measurement functions, approximating state and noise
 * with indii::ml::aux::GaussianPdf distributions. The advantage of
 * the RauchTungStriebelSmoother compared to KalmanSmoother is that
 * the measurements are not required for the backwards pass.
 * 
 * @see indii::ml::filter for general usage guidelines.
 * @see LinearModel for more detail on linear filters.
 */
template <class T>
class RauchTungStriebelSmoother : public Smoother<T> {
public:
  /**
   * Constructor.
   *
   * @param model Model to estimate.
   * @param tT \f$t_T\f$; starting time.
   * @param p_xT \f$p(\mathbf{x}_T)\f$; prior over the state at time
   * \f$t_T\f$.
   */
  RauchTungStriebelSmoother(RauchTungStriebelSmootherModel<T>* model,
      const T tT, const indii::ml::aux::GaussianPdf& p_xT);

  /**
   * Destructor.
   */
  virtual ~RauchTungStriebelSmoother();

  /**
   * Rewind system to time of previous measurement and
   * smooth.
   *
   * @param tn \f$t_n\f$; the time to which to rewind the
   * system. This must be less than the current time \f$t_{n+1}\f$.
   * @param p_xtn_ytn \f$p(\mathbf{x}_n\,|\,\mathbf{y}_{1:n})\f$; filter
   * density at time \f$t_n\f$.
   */
  virtual void smooth(const T tn,
      const indii::ml::aux::GaussianPdf& p_xtn_ytn);

  virtual indii::ml::aux::GaussianPdf smoothedMeasure();

private:
  /**
   * Model to estimate.
   */
  RauchTungStriebelSmootherModel<T>* model;

};

    }
  }
}

template <class T>
indii::ml::filter::RauchTungStriebelSmoother<T>::RauchTungStriebelSmoother(
    RauchTungStriebelSmootherModel<T>* model, const T tT,
    const indii::ml::aux::GaussianPdf& p_xT) : Smoother<T>(tT, p_xT),
    model(model) {
  //
}

template <class T>
indii::ml::filter::RauchTungStriebelSmoother<T>::~RauchTungStriebelSmoother() {
  //
}

template <class T>
void indii::ml::filter::RauchTungStriebelSmoother<T>::smooth(const T tn,
    const indii::ml::aux::GaussianPdf& p_xtn_ytn) {
  namespace aux = indii::ml::aux;

  /* pre-condition */
  assert (tn < this->tn);

  /* rewind time */
  T delta = this->tn - tn;
  this->tn = tn;

  /* calculate smoothed state for this time */
  aux::GaussianPdf p_xtnp1_ytn(model->p_xtnp1_ytn(p_xtn_ytn, delta));
  this->p_xtn_ytT = model->p_xtn_ytT(this->p_xtn_ytT, p_xtnp1_ytn,
      p_xtn_ytn, delta);

  /* post-condition */
  assert (this->tn == tn);
}

template <class T>
indii::ml::aux::GaussianPdf
    indii::ml::filter::RauchTungStriebelSmoother<T>::smoothedMeasure() {
  return model->p_y_x(this->p_xtn_ytT);
}

#endif

