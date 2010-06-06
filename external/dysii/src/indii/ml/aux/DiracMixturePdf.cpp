//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "DiracMixturePdf.hpp"
//#endif

#include "KDTreeNode.hpp"
#include "DistributedPartitioner.hpp"

#include "boost/mpi/operations.hpp"
#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_symmetric.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

#include <stack>

using namespace indii::ml::aux;

namespace ublas = boost::numeric::ublas;
namespace lapack = boost::numeric::bindings::lapack;

DiracMixturePdf::DiracMixturePdf() : MixturePdf<DiracPdf>(0), 
    Zsigma(0), haveSigma(false) {
  //
}

DiracMixturePdf::DiracMixturePdf(Pdf& o, const unsigned int P) :
    MixturePdf<DiracPdf>(o.getDimensions()), 
    Zsigma(o.getDimensions()), haveSigma(false) {
  unsigned int i;

  for (i = 0; i < P; i++) {
    add(o.sample());
  }
}

DiracMixturePdf::DiracMixturePdf(const unsigned int N) :
    MixturePdf<DiracPdf>(N), Zsigma(N), haveSigma(false) {
  //
}

DiracMixturePdf::~DiracMixturePdf() {
  //
}

double DiracMixturePdf::calculateEss() {
  const double W = getTotalWeight();
  const vector& ws = getWeights();
  double ess = 0.0;
  unsigned int i;
  
  for (i = 0; i < ws.size(); i++) {
    ess += ws(i)*ws(i);
  }
  ess = W*W / ess;

  return ess;
}

double DiracMixturePdf::calculateDistributedEss() {
  boost::mpi::communicator world;

  const double W = getDistributedTotalWeight();
  const vector& ws = getWeights();
  double ess = 0.0;
  unsigned int i;
  
  for (i = 0; i < ws.size(); i++) {
    ess += ws(i)*ws(i);
  }
  ess = W*W / boost::mpi::all_reduce(world, ess, std::plus<double>());

  return ess;
}

void DiracMixturePdf::standardise(const vector& mu,
    const lower_triangular_matrix& sd) {
  const unsigned int N = getDimensions();
  unsigned int i;

  /* prepare inverse standard deviation */
  matrix inv_sd(N,N), tmp(sd);
  inv(tmp, inv_sd);
  
  /* standardise */
  for (i = 0; i < getSize(); i++) {  
    get(i) = DiracPdf(prod(inv_sd, get(i) - mu));
  }

  dirty();    
}

void DiracMixturePdf::standardise(const vector& mu,
    const symmetric_matrix& sigma) {
  symmetric_matrix tmp(sigma);
  #ifdef NDEBUG
  lapack::pptrf(tmp); // avoids compiler warning about unused variable
  #else
  int err = lapack::pptrf(tmp);
  assert (err == 0);
  #endif
  
  lower_triangular_matrix sd(sigma.size1(), sigma.size2());
  noalias(sd) = ublas::triangular_adaptor<symmetric_matrix,ublas::lower>(tmp);
  standardise(mu, sd);
}

void DiracMixturePdf::distributedStandardise() {
  standardise(getDistributedExpectation(),
      getDistributedStandardDeviation());
}

void DiracMixturePdf::setDimensions(const unsigned int N,
    const bool preserve) {
  MixturePdf<DiracPdf>::setDimensions(N, preserve);

  Zsigma.resize(N, false);
}
  
symmetric_matrix& DiracMixturePdf::getCovariance()
{
    sigma = calculateCovariance(getExpectation());
    return sigma;
}

symmetric_matrix DiracMixturePdf::calculateCovariance(const vector& mu) {
  /* pre-condition */
  assert (getTotalWeight() > 0.0);
  unsigned int i;

  symmetric_matrix zsigma = zero_matrix(mu.size(), mu.size());
  for (i = 0; i < getSize(); i++) {
    if(getWeight(i) > 0)
        noalias(zsigma) += getWeight(i) * outer_prod(get(i)-mu, get(i)-mu);
  }
  return zsigma;
}

symmetric_matrix DiracMixturePdf::getDistributedCovariance() 
{
    boost::mpi::communicator world;
    const unsigned int size = world.size();

    /* If any Mean Changes, then all the covariances change */
    bool distHaveMu = boost::mpi::all_reduce(world, haveMu, std::logical_and<bool>());
    if(!distHaveMu) haveSigma = false;
    aux::vector mu = getDistributedExpectation();
    
    if (getTotalWeight() > 0.0) {
        if (!haveSigma) {
            Zsigma = calculateCovariance(mu);
        }
    } else {
        Zsigma.clear();
    }
    haveSigma = true;

    matrix tmp = boost::mpi::all_reduce(world, matrix(Zsigma),
                std::plus<matrix>())/getDistributedTotalWeight();
    return ublas::symmetric_adaptor<matrix, ublas::lower>(tmp);
}

lower_triangular_matrix DiracMixturePdf::getDistributedStandardDeviation() {
  boost::mpi::communicator world;
  const unsigned int size = world.size();
  lower_triangular_matrix sd_d(N,N);
  
  if (getDistributedSize() > 1) {
    symmetric_matrix sigma_d(getDistributedCovariance());
    int err;
    
    err = lapack::pptrf(sigma_d);
    assert (err == 0);
    noalias(sd_d) = ublas::triangular_adaptor<symmetric_matrix,
        ublas::lower>(sigma_d);
  } else {
    sd_d.clear();
  }

  return sd_d;
}

void DiracMixturePdf::redistributeBySpace() {
  boost::mpi::communicator world;
  unsigned int rank = world.rank();
  unsigned int size = world.size();
  
  unsigned int i, j;
  std::vector<unsigned int> is(getSize());

  for (i = 0; i < is.size(); i++) {
    is[i] = i;
  }

  /* build pruned kd tree of evenly distributed components */
  KDTreeNode* root = distributedBuild(is, 0, size);
  
  /* traverse tree and redistribute components */
  PartitionTreeNode* node = root;
  std::stack<PartitionTreeNode*> nodes;
  std::vector<PartitionTreeNode*> prunes;
  
  nodes.push(node);
  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node->isInternal()) {
      nodes.push(node->getLeft());
      nodes.push(node->getRight());
    } else if (node->isPrune()) {
      prunes.push_back(node);
    }
  }

  /* redistribute */
  std::vector<std::vector<DiracPdf> > recvXs;
  std::vector<vector> recvWeights;
  std::vector<DiracPdf> sendXs;
  vector sendWeights;

  for (i = 0; i < prunes.size(); i++) {
    /* prepare components at ith node in tree */
    is = prunes[i]->getIndices();
    sendXs.clear();
    sendWeights.resize(prunes[i]->getSize(), false);
    
    for (j = 0; j < is.size(); j++) {
      sendXs.push_back(get(is[j]));
      sendWeights(j) = getWeight(is[j]);
    }
    
    /* gather components to rank i */
    if (rank == i) {
      boost::mpi::gather(world, sendXs, recvXs, i);
      boost::mpi::gather(world, sendWeights, recvWeights, i);
    } else {
      boost::mpi::gather(world, sendXs, i);
      boost::mpi::gather(world, sendWeights, i);
    }
  }
  
  /* reconstruct */
  clear();
  for (i = 0; i < recvXs.size(); i++) {
    for (j = 0; j < recvXs[i].size(); j++) {
      add(recvXs[i][j], recvWeights[i](j));
    }
  }

  /* clean up */
  delete root;
}

void DiracMixturePdf::dirty() {
  MixturePdf<DiracPdf>::dirty();
  haveSigma = false;
}

KDTreeNode* DiracMixturePdf::distributedBuild(
    const std::vector<unsigned int>& is, const unsigned int depth,
    const unsigned int nodes) {
  /* pre-condition */
  assert (nodes > 0);
  
  boost::mpi::communicator world;
  
  KDTreeNode* result;
  unsigned int i, P, nth;
  
  if (nodes == 1) {
    /* prune node */
    result = new KDTreeNode(this, is, depth);
  } else {
    P = boost::mpi::all_reduce(world, is.size(), std::plus<unsigned int>());
    nth = P * (nodes / 2) / nodes;
    
    /* internal node */
    DistributedPartitioner partitioner(nth);
    KDTreeNode *left, *right; // child nodes
    std::vector<unsigned int> ls, rs; // indices of left, right components
    if (partitioner.init(this, is)) {
      ls.reserve(is.size() / 2);
      rs.reserve(is.size() / 2);
      
      for (i = 0; i < is.size(); i++) {
        if (partitioner.assign(get(is[i])) == Partitioner::LEFT) {
          ls.push_back(is[i]);
        } else {
          rs.push_back(is[i]);
        }
      }
      
      left = distributedBuild(ls, depth + 1, nodes / 2);
      right = distributedBuild(rs, depth + 1, nodes - nodes / 2);    

      result = new KDTreeNode(left, right, depth);
    } else {
      result = new KDTreeNode(this, is, depth);
    }
  }
  
  return result;
}
