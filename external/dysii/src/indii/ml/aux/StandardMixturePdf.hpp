#ifndef INDII_ML_AUX_STANDARDMIXTUREPDF_HPP
#define INDII_ML_AUX_STANDARDMIXTUREPDF_HPP

#include "MixturePdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Mixture probability density with standard calculation of covariance.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 564 $
 * @date $Date: 2008-09-12 16:35:34 +0100 (Fri, 12 Sep 2008) $
 *
 * @param T Component type, should be derivative of Pdf.
 *
 * @see MixturePdf for more information regarding the serialization
 * and parallelisation features of this class.
 */
template <class T>
class StandardMixturePdf : public MixturePdf<T> {
public:
  /**
   * Default constructor.
   *
   * Initialises the mixture with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization. Indeed, there is no other way to resize the
   * mixture to nonzero dimensionality except by subsequently
   * restoring from a serialization.
   */
  StandardMixturePdf();

  /**
   * Constructor. One or more components should be added with
   * add() after construction.
   *
   * @param N Dimensionality of the distribution.
   */
  StandardMixturePdf(const unsigned int N);

  /**
   * Constructor.
   *
   * @param x The first component.
   * @param w Unnormalised weight of the component.
   *
   * This is particularly useful for creating single component
   * mixtures of any type for parallel environments.
   */
  StandardMixturePdf(const T& x, const double w = 1.0);

  /**
   * Destructor.
   */
  virtual ~StandardMixturePdf();

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  /**
   * Get the covariance of the distribution. The covariance is defined
   * as:
   *
   * \f[ \frac{1}{W}\sum_{i=1}^{K} w^{(i)}(\Sigma^{(i)} +
   * \mathbf{\mu}^{(i)} (\mathbf{\mu}^{(i)})^{T}) -
   * \mathbf{\mu}\mathbf{\mu}^{T} \f]
   *
   * where \f$\mathbf{\mu}^{(i)}\f$, \f$\Sigma^{(i)}\f$ and
   * \f$w^{(i)}\f$ are the mean, covariance and weight of the
   * \f$i\f$th component, respectively, and \f$\mathbf{\mu}\f$ the
   * overall mean.
   *
   * @return \f$\Sigma\f$; covariance of the distribution.
   */
  virtual const symmetric_matrix& getCovariance();

  /**
   * Get the covariance of the full distribution.
   *
   * @return \f$\Sigma\f$; covariance of the full distribution.
   */
  symmetric_matrix getDistributedCovariance();

protected:
  virtual void dirty();

private:
  /**
   * Last calculated covariance.
   */
  symmetric_matrix sigma;

  /**
   * Last calculated unnormalized covariance.
   */
  symmetric_matrix Zsigma;

  /**
   * Is the last calculated covariance up to date?
   */
  bool haveSigma;

  /**
   * Calculate covariance from current components.
   *
   * The covariance is calculated as:
   *
   * \f[\Sigma =
   * \frac{1}{W}\sum_{i=1}^{K}w_i(\Sigma_i+\mathbf{\mu}_i\mathbf{\mu}_i^T)
   * - \bar{\mathbf{\mu}}\bar{\mathbf{\mu}}^T
   * \f]
   *
   * where \f$\mathbf{\mu}_i\f$ and \f$\Sigma_i\f$ are the mean and
   * covariance of the \f$i\f$th component, respectively, and
   * \f$\bar{\mathbf{\mu}}\f$ is the mean of means.
   */
  void calculateCovariance();

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

#include "boost/serialization/base_object.hpp"

namespace ublas = boost::numeric::ublas;

template <class T>
indii::ml::aux::StandardMixturePdf<T>::StandardMixturePdf() :
    indii::ml::aux::MixturePdf<T>(0), sigma(0), Zsigma(0) {
  haveSigma = false;
}

template <class T>
indii::ml::aux::StandardMixturePdf<T>::StandardMixturePdf(
    const unsigned int N) : indii::ml::aux::MixturePdf<T>(N), sigma(N),
    Zsigma(N) {
  haveSigma = false;
}

template <class T>
indii::ml::aux::StandardMixturePdf<T>::StandardMixturePdf(const T& x,
    const double w) : indii::ml::aux::MixturePdf<T>(x.getDimensions()),
    sigma(x.getDimensions()), Zsigma(x.getDimensions()) {
  haveSigma = false;
  add(x, w);
}

template <class T>
indii::ml::aux::StandardMixturePdf<T>::~StandardMixturePdf() {
  //
}

template <class T>
void indii::ml::aux::StandardMixturePdf<T>::setDimensions(
    const unsigned int N, const bool preserve) {
  MixturePdf<T>::setDimensions(N, preserve);

  Zsigma.resize(N, preserve);
  sigma.resize(N, preserve);
}

template <class T>
const indii::ml::aux::symmetric_matrix& indii::ml::aux::StandardMixturePdf<
    T>::getCovariance() {
  if (!haveSigma) {
    calculateCovariance();
  }
  return sigma;
}

template <class T>
indii::ml::aux::symmetric_matrix indii::ml::aux::StandardMixturePdf<
    T>::getDistributedCovariance() {
  boost::mpi::communicator world;
  const unsigned int size = world.size();
  
  if (size == 0) {
    return getCovariance();
  } else {
    const unsigned int N = this->getDimensions();

    boost::mpi::communicator world;
    matrix Zsigma_d(N,N), sigma_d(N,N);
    vector mu_d(N);

    if (this->getTotalWeight() > 0.0) {
      if (!haveSigma) {
        calculateCovariance();
      }
    } else {
      Zsigma.clear();
    }

    noalias(mu_d) = this->getDistributedExpectation();
    noalias(Zsigma_d) = boost::mpi::all_reduce(world, matrix(Zsigma),
        std::plus<matrix>());
    noalias(sigma_d) = Zsigma_d / this->getDistributedTotalWeight() -
        outer_prod(mu_d, mu_d);

    return ublas::symmetric_adaptor<matrix, ublas::lower>(sigma_d);
  }
}

template <class T>
void indii::ml::aux::StandardMixturePdf<T>::dirty() {
  MixturePdf<T>::dirty();
  haveSigma = false;
}

template <class T>
void indii::ml::aux::StandardMixturePdf<T>::calculateCovariance() {
  /* pre-condition */
  assert (this->getTotalWeight() > 0.0);

  double w;
  const vector& mu = this->getExpectation();
  unsigned int i;

  Zsigma.clear();
  for (i = 0; i < this->getSize(); i++) {
    w = this->getWeight(i);
    const vector& mu_i = this->get(i).getExpectation();
    const symmetric_matrix& sigma_i = this->get(i).getCovariance();

    noalias(Zsigma) += w * (sigma_i + outer_prod(mu_i, mu_i));
  }
  noalias(sigma) = Zsigma / this->getTotalWeight() - outer_prod(mu, mu);
  haveSigma = true;
}

template <class T>
template <class Archive>
void indii::ml::aux::StandardMixturePdf<T>::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<
      indii::ml::aux::MixturePdf<T> >(*this);
}

template <class T>
template <class Archive>
void indii::ml::aux::StandardMixturePdf<T>::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<
      indii::ml::aux::MixturePdf<T> >(*this);

  sigma.resize(this->getDimensions(), false);
  Zsigma.resize(this->getDimensions(), false);
  haveSigma = false;
}

#endif

