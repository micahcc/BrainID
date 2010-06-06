#ifndef INDII_ML_AUX_KERNELDENSITYPDF_HPP
#define INDII_ML_AUX_KERNELDENSITYPDF_HPP

#include "Pdf.hpp"
#include "PartitionTree.hpp"
#include "Almost2Norm.hpp"
#include "AlmostGaussianKernel.hpp"
#include "kde.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * %Kernel density estimator.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 584 $
 * @date $Date: 2008-12-15 17:26:36 +0000 (Mon, 15 Dec 2008) $
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 *
 * The kernel density estimator is constructed over a KDTree for efficient
 * evaluations. After construction, it acts as any other Pdf.
 *
 * @section Serialization
 *
 * This class supports serialization through the Boost.Serialization
 * library.
 *
 * @section KernelDensityPdf_references References
 * 
 * @anchor Silverman1986
 * Silverman, B.W. <i>Density Estimation for Statistics and Data
 * Analysis</i>. Chapman and Hall, <b>1986</b>.
 */
template <class NT = Almost2Norm, class KT = AlmostGaussianKernel>
class KernelDensityPdf : public Pdf {
public:
  /**
   * Default constructor.
   *
   * Initialises the distribution with zero dimensions. This should
   * generally only be used when the object is to be restored from a
   * serialization.
   */
  KernelDensityPdf();

  /**
   * Constructor.
   *
   * @param tree Partition tree over which to define distribution. Caller
   * has ownership.
   * @param N \f$\|\mathbf{x}\|_p\f$; a norm.
   * @param K \f$K(\|\mathbf{x}\|_p) \f$; density kernel.
   */
  KernelDensityPdf(PartitionTree* tree, const NT& N, const KT& K);
  
  /**
   * Destructor.
   */
  virtual ~KernelDensityPdf();

  /**
   * Not supported.
   *
   * @see Pdf::setDimensions()
   */
  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  virtual const vector& getExpectation();

  virtual const symmetric_matrix& getCovariance();

  virtual vector sample();

  virtual double densityAt(const vector& x);

  /**
   * Calculate the density for all points in a tree.
   *
   * @param tree Query tree.
   *
   * @return Density at all points in the tree, ordered according to the 
   * underlying DiracMixturePdf ordering.
   *
   * Uses a dual-tree algorithm to efficiently calculate the density at
   * all points in the query tree.
   */
  vector densityAt(PartitionTree& tree);

private:
  /**
   * Partition tree.
   */
  PartitionTree* tree;
  
  /**
   * \f$\|\mathbf{x}\|_p\f$; the norm.
   */
  NT N;
  
  /**
   * \f$K(\|\mathbf{x}\|_p) \f$; the density kernel.
   */
  KT K;
  
  /**
   * \f$\mathbf{\mu}\f$; the mean.
   */
  vector mu;
  
  /**
   * \f$\Sigma\f$; the covariance.
   */
  symmetric_matrix sigma;

  /**
   * Has the mean been calculated?
   */
  bool haveMu;
  
  /**
   * Has the covariance been calculated?
   */
  bool haveSigma;
  
  /**
   * Calculate the mean.
   */
  void calculateExpectation();
  
  /**
   * Calculate the covariance.
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

#include <stack>

template <class NT, class KT>
indii::ml::aux::KernelDensityPdf<NT,KT>::KernelDensityPdf() : tree(NULL),
    mu(0), sigma(0), haveMu(false), haveSigma(false) {
  //
}

template <class NT, class KT>
indii::ml::aux::KernelDensityPdf<NT,KT>::KernelDensityPdf(
    PartitionTree* tree, const NT& N, const KT& K) :
    Pdf(tree->getData()->getDimensions()), tree(tree), N(N), K(K),
    mu(tree->getData()->getDimensions()),
    sigma(tree->getData()->getDimensions()), haveMu(false), haveSigma(false) {
  //
}
  
template <class NT, class KT>
indii::ml::aux::KernelDensityPdf<NT,KT>::~KernelDensityPdf() {
  //
}

template <class NT, class KT>
void indii::ml::aux::KernelDensityPdf<NT,KT>::setDimensions(
    const unsigned int N, const bool preserve) {
  assert (false);
}

template <class NT, class KT>
const indii::ml::aux::vector&
    indii::ml::aux::KernelDensityPdf<NT,KT>::getExpectation() {
  if (!haveMu) {
    calculateExpectation();
    assert (haveMu);
  }
  return mu;
}

template <class NT, class KT>
const indii::ml::aux::symmetric_matrix&
    indii::ml::aux::KernelDensityPdf<NT,KT>::getCovariance() {
  if (!haveSigma) {
    calculateCovariance();
    assert (haveSigma);
  }
  return sigma;
}

template <class NT, class KT>
indii::ml::aux::vector indii::ml::aux::KernelDensityPdf<NT,KT>::sample() {
  /* pre-condition */
  assert (tree != NULL);

  /* sampling from the underlying weighted sample set and adding a kernel
   * sample is more efficient than working down the tree. */
  return tree->getData()->sample() + K.sample() * N.sample(getDimensions());
}

template <class NT, class KT>
double indii::ml::aux::KernelDensityPdf<NT,KT>::densityAt(const vector& x) {
  if (tree->getRoot() == NULL) {
    return 0.0;
  }

  PartitionTreeNode* node = tree->getRoot();
  DiracMixturePdf* p = tree->getData();
  std::stack<PartitionTreeNode*> nodes;
  vector d(x.size());
  double result = 0.0;
  std::vector<unsigned int> is;
  unsigned int i;

  nodes.push(node);
  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();
    
    if (node->isLeaf()) {
      i = node->getIndex();
      noalias(d) = p->get(i) - x;
      result += p->getWeight(i) * K(N(d));
    } else if (node->isPrune()) {
      is = node->getIndices();
      for (i = 0; i < is.size(); i++) {
        noalias(d) = p->get(is[i]) - x;
        result += p->getWeight(is[i]) * K(N(d));
      }
    } else {
      /* should we recurse? */
      node->difference(x, d);
      if (K(N(d)) > 0.0) {
        nodes.push(node->getLeft());
        nodes.push(node->getRight());
      } 
    }
  }  
  return result / p->getTotalWeight();
}

template <class NT, class KT>
indii::ml::aux::vector
    indii::ml::aux::KernelDensityPdf<NT,KT>::densityAt(PartitionTree& tree) {
  /* pre-condition */
  assert (getDimensions() == tree.getData()->getDimensions());

  return aux::dualTreeDensity(tree, *this->tree,
      this->tree->getData()->getWeights(), N, K);
}

template <class NT, class KT>
void indii::ml::aux::KernelDensityPdf<NT,KT>::calculateExpectation() {
  std::vector<PartitionTreeNode*> nodes;
  PartitionTreeNode* node = tree->getRoot();
  DiracMixturePdf* p = tree->getData();
  assert (node != NULL);

  std::vector<unsigned int> is;
  unsigned int i;

  mu.clear();
  nodes.push_back(node);
  while (!nodes.empty()) {
    node = nodes.back();
    nodes.pop_back();
    if (node->isLeaf()) {
      i = node->getIndex();
      mu += p->getWeight(i) * p->get(i);
    } else if (node->isPrune()) {
      is = node->getIndices();
      for (i = 0; i < is.size(); i++) {
        mu += p->getWeight(is[i]) * p->get(is[i]);
      }
    } else {
      /* recurse */
      nodes.push_back(node->getLeft());
      nodes.push_back(node->getRight());
    }
  }
  
  mu /= p->getTotalWeight();
  haveMu = true;
}

template <class NT, class KT>
void indii::ml::aux::KernelDensityPdf<NT,KT>::calculateCovariance() {
  if (!haveMu) {
    calculateExpectation();
    assert (haveMu);
  }
 
  std::vector<PartitionTreeNode*> nodes;
  PartitionTreeNode* node = tree->getRoot();
  DiracMixturePdf* p = tree->getData();
  assert (node != NULL);

  std::vector<unsigned int> is;
  unsigned int i;

  sigma.clear();
  nodes.push_back(node);
  while (!nodes.empty()) {
    node = nodes.back();
    nodes.pop_back();

    if (node->isLeaf()) {
      i = node->getIndex();
      vector& x = p->get(i);
      sigma += p->getWeight(i) * outer_prod(x, x);
    } else if (node->isPrune()) {
      is = node->getIndices();
      for (i = 0; i < is.size(); i++) {
        vector& x = p->get(is[i]);
        sigma += p->getWeight(is[i]) * outer_prod(x, x);
      }
    } else {
      /* recurse */
      nodes.push_back(node->getLeft());
      nodes.push_back(node->getRight());
    }
  }  
  sigma /= p->getTotalWeight();
  noalias(sigma) -= outer_prod(mu, mu);
  haveSigma = true;
}

template <class NT, class KT>
template <class Archive>
void indii::ml::aux::KernelDensityPdf<NT,KT>::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<Pdf>(*this);
  ar & tree;
  ar & N;
  ar & K;
}

template <class NT, class KT>
template <class Archive>
void indii::ml::aux::KernelDensityPdf<NT,KT>::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<Pdf>(*this);
  ar & tree;
  ar & N;
  ar & K;

  mu.resize(tree->getData()->getDimensions(), false);
  sigma.resize(tree->getData()->getDimensions(), false);
  haveMu = false;
  haveSigma = false;
}

#endif

