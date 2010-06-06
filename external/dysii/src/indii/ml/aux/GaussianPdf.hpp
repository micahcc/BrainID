#ifndef INDII_ML_AUX_GAUSSIANPDF_HPP
#define INDII_ML_AUX_GAUSSIANPDF_HPP

#include "Pdf.hpp"

#include "boost/serialization/split_member.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Multivariate Gaussian probability distribution.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 538 $
 * @date $Date: 2008-08-31 14:41:10 +0100 (Sun, 31 Aug 2008) $
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library. Two issues are worth noting.
 *
 * Firstly, Boost.uBLAS has limited serialization support in its CVS
 * snapshot at time of writing (3 Sep 2007). It appears limited to the
 * serialization of basic vectors and matrices: symmetric matrices do
 * not appear to be supported, for example.
 *
 * Secondly, it's arguable whether serializing pre-calculations is
 * useful or not. It makes for larger messages in a parallel
 * environment, the cost of sending which may exceed the savings of
 * the precalculations in the first place.
 *
 * The implementation here takes the path of least resistance in both
 * cases. We choose to use the limited Boost.uBLAS serialization
 * features anyway for ease of implementation, presuming that
 * eventually full serialization support will be provided. We also
 * choose to serialize just the bare essentials, excluding
 * pre-calculation results.
 */
class GaussianPdf : public Pdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the Gaussian with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  GaussianPdf();

  /**
   * Construct Gaussian given mean and covariance.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean of the Gaussian.
   * @param sigma \f$\Sigma\f$; covariance of the Gaussian.
   */
  GaussianPdf(const vector& mu, const symmetric_matrix& sigma);

  /**
   * Construct Gaussian given dimensionality. The mean and covariance
   * remain uninitialised, and should be set with setExpectation() and
   * setCovariance().
   *
   * @param N \f$N\f$; number of dimensions of the Gaussian.
   */
  GaussianPdf(unsigned int N);

  /**
   * Assignment operator. Both sides must have the same dimensionality.
   */
  GaussianPdf& operator=(const GaussianPdf& o);

  /**
   * Destructor.
   */
  virtual ~GaussianPdf();

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  /**
   * Get the expected value of the distribution.
   *
   * @return \f$\mathbf{\mu}\f$; expected value of the distribution.
   */
  virtual const vector& getExpectation() const;

  /**
   * Get the covariance of the distribution.
   *
   * @return \f$\Sigma\f$; covariance of the distribution.
   */
  virtual const symmetric_matrix& getCovariance() const;

  virtual const vector& getExpectation();

  virtual const symmetric_matrix& getCovariance();

  /**
   * Set the mean of the Gaussian. The new mean vector must have the
   * same size as the previous one.
   *
   * @param mu \f$\mathbf{\mu}\f$; mean of the Gaussian.
   */
  virtual void setExpectation(const vector& mu);

  /**
   * Set the covariance of the Gaussian. The new covariance matrix
   * must have the same size as the previous one.
   *
   * @param sigma \f$\Sigma\f$; covariance of the Gaussian.
   */
  virtual void setCovariance(const symmetric_matrix& sigma);

  /**
   * Sample from the Gaussian.
   *
   * @li Let \f$\mathbf{z}\f$ be a vector of \f$N\f$ independent
   * normal variates.
   * @li Then the sample is \f$\mathbf{x} = \mathbf{\mu} +
   * L\mathbf{z}\f$, where \f$L\f$ is the Cholesky decomposition of
   * the covariance matrix \f$\Sigma\f$.
   *
   * @return The sample \f$\mathbf{x}\f$
   */
  virtual vector sample();

  /**
   * @deprecated Use densityAt() instead.
   */
  double calculateDensity(const vector& x);

  /**
   * Calculate the density at a given point.
   *
   * \f[p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^N|\Sigma|}}
   *   \exp\big({-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^T \Sigma^{-1} 
   *   (\mathbf{x}-\mathbf{\mu})\big)}
   * \f]
   *
   * @param x \f$\mathbf{x}\f$; point at which to evaluate the
   * density.
   *
   * @return \f$p(\mathbf{x})\f$; the density at \f$\mathbf{x}\f$.
   */
  virtual double densityAt(const vector& x);

protected:
  /**
   * Called when changes are made to the distribution, such as when
   * the expectation or covariance is changed. This allows
   * pre-calculations to be refreshed.
   */
  virtual void dirty();

private:
  /**
   * \f$\mu\f$; mean of the distribution.
   */
  vector mu;

  /**
   * \f$\Sigma\f$; covariance of the distribution.
   */
  symmetric_matrix sigma;

  /**
   * \f$L\f$ where \f$\Sigma = LL^T\f$ is the Cholesky decomposition
   * of the covariance matrix.
   */
  lower_triangular_matrix L;

  /**
   * \f$\Sigma^{-1}\f$
   */
  symmetric_matrix sigmaI;

  /**
   * Diagonal of \f$\Sigma^{-1}\f$.
   */
  vector sigmaIDiag;

  /**
   * \f$\det(\Sigma)\f$
   */
  double sigmaDet;

  /**
   * \f$\frac{1}{Z}\f$
   */
  double ZI;

  /**
   * Is the expectation zero?
   */
  bool isMuZero;

  /**
   * Is the covariance matrix diagonal?
   */
  bool isSigmaIDiagonal;

  /**
   * Has the Cholesky decomposition of the covariance matrix been computed?
   */
  bool haveCholesky;

  /**
   * Has the inverse of the covariance matrix been computed?
   */
  bool haveInverse;

  /**
   * Has the determinant of the covariance matrix been computed?
   */
  bool haveDeterminant;

  /**
   * Has the normalising constant for the density function been
   * computed?
   */
  bool haveZ;

  /**
   * Have optimisations for density calculations been determined?
   */
  bool haveDensityOptimisations;

  /**
   * Calculate the Cholesky decomposition of the covariance matrix.
   */
  void calculateCholesky();

  /**
   * Calculate the inverse of the covariance matrix. This exploits the
   * calculation of the Cholesky decomposition
   */
  void calculateInverse();

  /**
   * Calculate the determinant of the covariance matrix,
   * \f$|\Sigma|\f$. This exploits the calculation of the Cholesky
   * decomposition \f$L\f$. For any matrices \f$A\f$ and \f$B\f$,
   * \f$|AB| = |A||B|\f$. Given that \f$\Sigma = LL^T\f$, \f$|\Sigma|
   * = |L|^2\f$. The determinant of a triangular matrix is simply the
   * product of the elements on its diagonal, so \f$|L|\f$ is easy to
   * calculate.
   */
  void calculateDeterminant();

  /**
   * Calculate the normalising constant \f$\frac{1}{Z}\f$ for the
   * density function.
   *
   * \f[Z = \sqrt{(2\pi)^N|\Sigma|}\f]
   */
  void calculateZ();

  /**
   * Calculate optimisations for the calculation of densities.
   */
  void calculateDensityOptimisations();

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

template<class Archive>
void indii::ml::aux::GaussianPdf::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<Pdf>(*this);
  ar & mu;

  /* serialization of symmetric_matrix not yet supported in ublas */
  const aux::matrix tmp(sigma);
  ar & tmp;
}

template<class Archive>
void indii::ml::aux::GaussianPdf::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<Pdf>(*this);

  mu.resize(N, false);
  sigma.resize(N, false);
  L.resize(N,N, false);
  sigmaI.resize(N, false);
  sigmaIDiag.resize(N, false);

  ar & mu;

  /* serialization of symmetric_matrix not yet supported in ublas */
  aux::matrix tmp(N,N);
  ar & tmp;
  noalias(sigma) = boost::numeric::ublas::symmetric_adaptor<aux::matrix>(tmp);

  haveCholesky = false;
  haveInverse = false;
  haveDeterminant = false;
  haveZ = false;
  haveDensityOptimisations = false;
}

#endif
