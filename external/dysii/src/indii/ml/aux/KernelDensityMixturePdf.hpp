#ifndef INDII_ML_AUX_KERNELDENSITYMIXTUREPDF_HPP
#define INDII_ML_AUX_KERNELDENSITYMIXTUREPDF_HPP

#include "StandardMixturePdf.hpp"
#include "KernelDensityPdf.hpp"
#include "KDTree.hpp"
#include "Almost2Norm.hpp"
#include "AlmostGaussianKernel.hpp"

namespace indii {
  namespace ml {
    namespace aux {
/**
 * Mixture of kernel density estimators.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 590 $
 * @date $Date: 2008-12-17 15:09:40 +0000 (Wed, 17 Dec 2008) $
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 *
 * @see MixturePdf for more information regarding the serialization
 * and parallelisation features of this class.
 */
template <class NT = Almost2Norm, class KT = AlmostGaussianKernel>
class KernelDensityMixturePdf :
    public StandardMixturePdf<KernelDensityPdf<NT,KT> > {
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
  KernelDensityMixturePdf();

  /**
   * Constructor. One or more components should be added with
   * addComponent() after construction.
   *
   * @param N Dimensionality of the distribution.
   */
  KernelDensityMixturePdf(const unsigned int N);

  /**
   * Constructor.
   *
   * @param x The first component.
   * @param w Unnormalised weight of the component.
   *
   * This is particularly useful for creating single component
   * mixtures of any type for parallel environments.
   */
  KernelDensityMixturePdf(const KernelDensityPdf<NT,KT>& x,
      const double w = 1.0);

  /**
   * Destructor.
   */
  virtual ~KernelDensityMixturePdf();

  using StandardMixturePdf<KernelDensityPdf<NT,KT> >::densityAt;
  
  using StandardMixturePdf<KernelDensityPdf<NT,KT> >::distributedDensityAt;

  /**
   * Calculate the density on the local node for all points in a tree.
   *
   * @param tree Query tree.
   *
   * @return Density at all points in the tree, ordered according to the 
   * underlying DiracMixturePdf ordering.
   *
   * Uses a dual-tree algorithm to efficiently calculate the density at
   * all points in the query tree.
   *
   * @todo Currently requires KDTree rather than PartitionTree due to
   * apparent link errors related to Boost.MPI and Boost.Serialization.
   */
  vector densityAt(PartitionTree& tree);

  /**
   * Calculate the density of the full distribution for all points in a
   * tree.
   *
   * @param tree Query tree on this node.
   *
   * @return Density at all points in the tree on this node, ordered
   * according to its underlying DiracMixturePdf ordering.
   *
   * Uses a dual-tree algorithm to efficiently calculate the density at
   * all points in the query tree. Note that while each node is passed only
   * its set of points, in the form of a tree, and returns only the density
   * calculations for that set of points, all nodes participate in the
   * calculation for all points.
   *
   * @todo Currently requires KDTree rather than PartitionTree due to
   * apparent link errors related to Boost.MPI and Boost.Serialization.
   */
  template <class S>
  vector distributedDensityAt(KDTree<S>& tree);

private:
  /**
   * Serialize or restore from serialization.
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

#include "boost/serialization/base_object.hpp"

template <class NT, class KT>
indii::ml::aux::KernelDensityMixturePdf<NT,KT>::KernelDensityMixturePdf() :
    StandardMixturePdf<KernelDensityPdf<NT,KT> >() {
  //
}

template <class NT, class KT>
indii::ml::aux::KernelDensityMixturePdf<NT,KT>::KernelDensityMixturePdf(
    const unsigned int N) : StandardMixturePdf<KernelDensityPdf<NT,KT> >(N) {
  //
}

template <class NT, class KT>
indii::ml::aux::KernelDensityMixturePdf<NT,KT>::KernelDensityMixturePdf(
    const KernelDensityPdf<NT,KT>& x, const double w) :
    indii::ml::aux::StandardMixturePdf<KernelDensityPdf<NT,KT> >(x, w) {
  //
}

template <class NT, class KT>
indii::ml::aux::KernelDensityMixturePdf<NT,KT>::~KernelDensityMixturePdf() {
  //
}

template <class NT, class KT>
indii::ml::aux::vector
    indii::ml::aux::KernelDensityMixturePdf<NT,KT>::densityAt(
    PartitionTree& tree) {
  vector result(tree.getRoot()->getSize());
  unsigned int i;
  
  result.clear();
  for (i = 0; i < this->getSize(); i++) {
    noalias(result) += this->getWeight(i) * this->get(i).densityAt(tree);
  }
  result /= this->getTotalWeight();
  
  return result;
}

template <class NT, class KT>
template <class S>
indii::ml::aux::vector
    indii::ml::aux::KernelDensityMixturePdf<NT,KT>::distributedDensityAt(
    KDTree<S>& tree) {
  boost::mpi::communicator world;
  const unsigned int size = world.size();
  unsigned int i;
 
  vector result(this->getTotalWeight() * this->densityAt(tree));
  rotate(*tree.getData());
  rotate(tree);
  rotate(result);

  for (i = 1; i < size; i++) {
    noalias(result) += this->getTotalWeight() * this->densityAt(tree);
    rotate(*tree.getData());
    rotate(tree);
    rotate(result);
  }
  result /= this->getDistributedTotalWeight();

  return result;
}

template <class NT, class KT>
template <class Archive>
void indii::ml::aux::KernelDensityMixturePdf<NT,KT>::serialize(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<
      indii::ml::aux::StandardMixturePdf<
      indii::ml::aux::KernelDensityPdf<NT,KT> > >(*this);
}

#endif

