#ifndef INDII_ML_FILTER_FILTER_HPP
#define INDII_ML_FILTER_FILTER_HPP

#include "../aux/vector.hpp"
#include "../aux/GaussianPdf.hpp"
#include "boost/serialization/serialization.hpp"

namespace indii {
  namespace ml {
    namespace filter {
/**
 * Abstract %filter.
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
class Filter {
public:
  /**
   * Constructor.
   *
   * @param p_x0 \f$p(\mathbf{x}_0)\f$; prior over the initial state
   * \f$\mathbf{x}_0\f$.
   */
  Filter(const P& p_x0);

  /**
   * Destructor.
   */
  virtual ~Filter();

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
   * Get filter density.
   *
   * @return \f$p(\mathbf{x}(t_n)\,|\,\mathbf{y}_{1:n}\f$; the filter
   * density at the current time.
   */
  P& getFilteredState();

  /**
   * Set the filter density.
   *
   * @return \f$p(\mathbf{x}(t_n)\,|\,\mathbf{y}_{1:n}\f$; the filter
   * density at the current time.
   */
  void setFilteredState(const P& p_xtn_ytn);

  /**
   * Advance system to time of next measurement.
   *
   * @param tnp1 \f$t_{n+1}\f$; the time to which to advance the
   * system. This must be greater than the current time \f$t_n\f$.
   * @param ytnp1 \f$\mathbf{y}_{n+1}\f$; measurement at time
   * \f$t_{n+1}\f$.
   */
  virtual void filter(const T tnp1, const indii::ml::aux::vector& ytnp1)
      = 0;

  /**
   * Apply the measurement function to the current filtered state to
   * obtain an estimated measurement.
   *
   * @return The estimated measurement.
   */
  virtual P measure() = 0;

protected:
  /**
   * \f$t_n\f$; the current time. For internal use only.
   */
  T tn;

  /**
   * \f$p(\mathbf{x}(t_n)\,|\,\mathbf{y}_{1:n}\f$; the filter
   * density at the current time.
   */
  P p_xtn_ytn;

  /**
   * Serialization
   */
  template< class Archive >
  void serialize(Archive & ar, unsigned int verison)
  {
        ar & tn;
        ar & p_xtn_ytn;
  }
  friend class boost::serialization::access;
};

    }
  }
}

template <class T, class P>
indii::ml::filter::Filter<T,P>::Filter(const P& p_x0) : p_xtn_ytn(p_x0) {
  this->tn = 0;
}

template <class T, class P>
indii::ml::filter::Filter<T,P>::~Filter() {
  //
}

template <class T, class P>
inline T indii::ml::filter::Filter<T,P>::getTime() const {
  return this->tn;
}

template <class T, class P>
void indii::ml::filter::Filter<T,P>::setTime(const T tn) {
  this->tn = tn;
}

template <class T, class P>
inline P& indii::ml::filter::Filter<T,P>::getFilteredState() {
  return this->p_xtn_ytn;
}

template <class T, class P>
void indii::ml::filter::Filter<T,P>::setFilteredState(const P& p_xtn_ytn) {
  this->p_xtn_ytn = p_xtn_ytn;
}

#endif

