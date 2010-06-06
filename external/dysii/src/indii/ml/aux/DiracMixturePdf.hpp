#ifndef INDII_ML_AUX_DIRACMIXTUREPDF_HPP
#define INDII_ML_AUX_DIRACMIXTUREPDF_HPP

#include "MixturePdf.hpp"
#include "DiracPdf.hpp"

namespace indii {
  namespace ml {
    namespace aux {
    
      class KDTreeNode;
    
/**
 * Dirac mixture probability density.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 538 $
 * @date $Date: 2008-08-31 14:41:10 +0100 (Sun, 31 Aug 2008) $
 *
 * @see MixturePdf for more information regarding the serialization
 * and parallelisation features of this class.
 */
class DiracMixturePdf : public MixturePdf<DiracPdf> {
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
  DiracMixturePdf();

  /**
   * Sampling constructor.
   *
   * @param o Distribution to approximate.
   * @param P Number of samples with which to approximate the
   * distribution.
   * 
   * Initialises the mixture by drawing @c P equally weighted samples
   * from the distribution @c o.
   */
  DiracMixturePdf(Pdf& o, const unsigned int P);

  /**
   * Constructor. One or more components should be added with
   * addComponent() after construction.
   *
   * @param N Dimensionality of the distribution.
   */
  DiracMixturePdf(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~DiracMixturePdf();

  /**
   * Calculate effective sample size on the local node.
   *
   * @return Effective sample size of the components on the local node.
   */
  double calculateEss();
  
  /**
   * Calculate effective sample size of the full distribution.
   *
   * @return Effective sample size of the full distribution.
   */
  double calculateDistributedEss();

  /**
   * Standardise components on the local node to zero mean and identity
   * covariance.
   */
  virtual void standardise();

  /**
   * Standardise components on the local node using given mean and 
   * standard deviation.
   *
   * @param mu Mean to use for standardisation.
   * @param sd Standard deviation to use for standardisation.
   */
  virtual void standardise(const vector& mu,
      const lower_triangular_matrix& sd);

  /**
   * Standardise components on the local node using given mean and 
   * covariance.
   *
   * @param mu Mean to use for standardisation.
   * @param sigma Covariance to use for standardisation.
   */
  virtual void standardise(const vector& mu, const symmetric_matrix& sigma);

  /**
   * Standardise components across all nodes to zero mean and identity
   * covariance.
   */
  virtual void distributedStandardise();

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  virtual const symmetric_matrix& getCovariance();

  /**
   * Get the covariance of the full distribution.
   *
   * @return \f$\Sigma\f$; covariance of the full distribution.
   */
  symmetric_matrix getDistributedCovariance();

  /**
   * Get the standard deviation of the distribution.
   *
   * @return \f$\Sigma^{1/2}\f$; standard deviation of the distribution.
   */
  lower_triangular_matrix getStandardDeviation();

  /**
   * Get the standard deviation of the full distribution.
   *
   * @return \f$\Sigma^{1/2}\f$; standard deviation of the full
   * distribution.
   */
  lower_triangular_matrix getDistributedStandardDeviation();

  /**
   * Redistribute components across nodes by space. This builds a shallow
   * \f$kd\f$ tree distributed across all nodes such that the number of
   * leaves equals the number of nodes in the parallel environment. Using 
   * \f$n\f$th element splits, the algorithm ensures the same number of 
   * samples in each leaf node. Each node in the parallel environment then
   * adopts all the samples at one of these leaf nodes. In this way, there
   * is some spatial locality to all the samples on one node, ideal for 
   * further building of a distributed mixture of \f$kd\f$ trees using
   * KernelDensityMixturePdf.
   */
  void redistributeBySpace();

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
   */
  void calculateCovariance();

  /**
   * Build kd tree node for redistributeBySpace().
   *
   * @param is Indices of components over which to build tree.
   * @param depth Depth of the node in the tree.
   * @param nodes Number of nodes over which to distribute components at
   * this node.
   */
  KDTreeNode* distributedBuild(const std::vector<unsigned int>& is,
      const unsigned int depth, const unsigned int nodes);

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

template <class Archive>
void indii::ml::aux::DiracMixturePdf::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<MixturePdf<DiracPdf> >(*this);
}

template <class Archive>
void indii::ml::aux::DiracMixturePdf::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<MixturePdf<DiracPdf> >(*this);

  sigma.resize(getDimensions(), false);
  Zsigma.resize(getDimensions(), false);
  haveSigma = false;
}

#endif

