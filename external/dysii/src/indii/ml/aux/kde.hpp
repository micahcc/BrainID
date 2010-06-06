#ifndef INDII_ML_AUX_KDE_HPP
#define INDII_ML_AUX_KDE_HPP

#include "PartitionTree.hpp"

/**
 * @file kde.hpp
 *
 * Provides convenience methods for working with kernel density
 * approximations.
 */

namespace indii {
  namespace ml {
    namespace aux {

/**
 * Calculate \f$h_{opt}\f$.
 *
 * @param N Number of dimensions.
 * @param P Number of samples.
 *
 * Note:
 *
 * \f[
 * h_{opt} = \left[\frac{4}{(N+2)P}\right]^{\frac{1}{N+4}}\,,
 * \f]
 *
 * this being the optimal bandwidth for a kernel density estimate
 * over \f$P\f$ samples from a standard \f$N\f$-dimensional Gaussian
 * distribution, and Gaussian kernel (@ref Silverman1986
 * "Silverman, 1986"). We find this useful as a rule of thumb for
 * setting kernel density estimate bandwidths.
 *
 * @section hopt_references References
 *
 * @anchor Silverman1986
 * Silverman, B.W. <i>Density Estimation for Statistics and Data
 * Analysis</i>. Chapman and Hall, <b>1986</b>.
 */
double hopt(const unsigned int N, const unsigned int P);

/**
 * Dual-tree kernel density evaluation.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param queryTree Query tree.
 * @param targetTree Target tree.
 * @param w Weight vector.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Vector of the density estimates for each of the points in 
 * queryTree, ordered according to its underlying data.
 */
template <class NT, class KT, class PT>
vector dualTreeDensity(PT& queryTree, PT& targetTree,
    const vector& w, const NT& N, const KT& K, const bool normalise = true);
      
/**
 * Distributed dual-tree kernel density evaluation.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param queryTree Query tree.
 * @param targetTree Target tree.
 * @param w Weight vector.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Vector of the density estimates for each of the points in 
 * queryTree, ordered according to its underlying data. Note that while
 * only the results for the local query components are returned, all
 * nodes participate in the evaluation.
 *
 * Note that queryTree and targetTree should have different underlying
 * DiracMixturePdf objects.
 */
template <class NT, class KT, class PT>
vector distributedDualTreeDensity(PT& queryTree,
    PT& targetTree, const vector& w, const NT& N, const KT& K,
    const bool normalise = true);

/**
 * Self-tree kernel density evaluation.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param tree Tree.
 * @param w Weight vector.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Vector of the density estimates for each of the points in 
 * queryTree, ordered according to its underlying data.
 */
template <class NT, class KT, class PT>
vector selfTreeDensity(PT& tree, const vector& w, const NT& N,
    const KT& K, const bool normalise = true);
      
/**
 * Distributed self-tree kernel density evaluation.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param tree Tree.
 * @param w Weight vector.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Vector of the density estimates for each of the points in 
 * queryTree, ordered according to its underlying data. Note that while
 * only the results for the local query components are returned, all
 * nodes participate in the evaluation.
 */
template <class NT, class KT, class PT>
vector distributedSelfTreeDensity(PT& tree, const vector& w,
    const NT& N, const KT& K, const bool normalise = true);

/**
 * Dual-tree kernel density evaluation with multiple mixture model
 * weights.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param queryTree Query tree.
 * @param targetTree Target tree.
 * @param ws Weight matrix. Each row gives a weight vector for one mixture
 * model.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Matrix where each row provides the density estimates for each
 * of the points in queryTree, ordered according its underlying data, and
 * using the weights of the corresponding row in ws. The weights
 * underlying queryTree and targetTree are ignored. Optimisations are
 * made in the case that the queryTree and targetTree are identical and
 * over the same underlying data.
 */
template <class NT, class KT, class PT>
matrix dualTreeDensity(PT& queryTree, PT& targetTree,
    const matrix& ws, const NT& N, const KT& K, const bool normalise = true);
      
/**
 * Distributed dual-tree kernel density evaluation with multiple mixture
 * model weights.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param queryTree Query tree.
 * @param targetTree Target tree.
 * @param ws Weight matrix. Each row gives a weight vector for one mixture
 * model.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Matrix where each row provides the density estimates for each
 * of the points in queryTree, ordered according its underlying data, and
 * using the weights of the corresponding row in ws. The weights
 * underlying queryTree and targetTree are ignored.
 *
 * Note that queryTree and targetTree should have different underlying
 * DiracMixturePdf objects.
 */
template <class NT, class KT, class PT>
matrix distributedDualTreeDensity(PT& queryTree,
    PT& targetTree, const matrix& ws, const NT& N, const KT& K,
    const bool normalise = true);

/**
 * Self-tree kernel density evaluation with multiple mixture model
 * weights.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param tree Tree.
 * @param ws Weight matrix. Each row gives a weight vector for one mixture
 * model.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Matrix where each row provides the density estimates for each
 * of the points in queryTree, ordered according its underlying data, and
 * using the weights of the corresponding row in ws. The weights underlying
 * tree are ignored.
 */
template <class NT, class KT, class PT>
matrix selfTreeDensity(PT& tree, const matrix& ws, const NT& N,
    const KT& K, const bool normalise = true);
      
/**
 * Distributed self tree kernel density evaluation with multiple mixture
 * model weights.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param tree Tree.
 * @param ws Weight matrix. Each row gives a weight vector for one mixture
 * model.
 * @param N Norm.
 * @param K Kernel.
 * @param normalise Normalise results.
 *
 * @return Matrix where each row provides the density estimates for each
 * of the points in queryTree, ordered according its underlying data, and
 * using the weights of the corresponding row in ws. The weights underlying
 * tree are ignored.
 */
template <class NT, class KT, class PT>
matrix distributedSelfTreeDensity(PT& tree, const matrix& ws,
    const NT& N, const KT& K, const bool normalise = true);

/**
 * Cross-tree kernel density evaluation with multiple mixture model
 * weights. Simultaneously performs kernel density estimation of two 
 * trees to each other.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 * @param PT Partition tree type.
 *
 * @param tree1 First tree.
 * @param tree2 Second tree.
 * @param ws1 Weight matrix for first tree as target. Each row gives a
 * weight vector for one mixture model.
 * @param ws2 Weight matrix for second tree as target. Each row gives a
 * weight vector for one mixture model.
 * @param N Norm.
 * @param K Kernel.
 * @param result1 On return, unnormalised result for first tree as query,
 * second as target, is added to this.
 * @param result2 On return, unnormalised result for second tree as query,
 * first as target, is added to this.
 * @param clear Clear result matrices before addition.
 * @param normalise Normalise results.
 *
 * Intended mainly for internal use.
 */
template <class NT, class KT, class PT>
void crossTreeDensity(PT& tree1, PT& tree2,
    const matrix& ws1, const matrix& ws2, const NT& N, const KT& K,
    matrix& result1, matrix& result2, const bool clear = true,
    const bool normalise = true);

    }
  }
}

#include "DiracMixturePdf.hpp"
#include "KDTree.hpp"
#include "vector.hpp"
#include "matrix.hpp"

#include <stack>

inline double indii::ml::aux::hopt(const unsigned int N,
    const unsigned int P) {
  return pow(4.0/((N+2)*P), 1.0/(N+4.0));
}

template <class NT, class KT, class PT>
indii::ml::aux::vector indii::ml::aux::dualTreeDensity(
    PT& queryTree, PT& targetTree, const vector& w,
    const NT& N, const KT& K, const bool normalise) {
  aux::matrix ws(1,w.size());
  row(ws,0) = w;
  return row(dualTreeDensity(queryTree, targetTree, ws, N, K, normalise), 0);
}

template <class NT, class KT, class PT>
indii::ml::aux::vector indii::ml::aux::distributedDualTreeDensity(
    PT& queryTree, PT& targetTree, const vector& w,
    const NT& N, const KT& K, const bool normalise) {
  aux::matrix ws(1,w.size());
  row(ws,0) = w;
  return row(distributedDualTreeDensity(queryTree, targetTree, ws, N, K,
      normalise), 0);
}

template <class NT, class KT, class PT>
indii::ml::aux::vector indii::ml::aux::selfTreeDensity(
    PT& tree, const vector& w, const NT& N, const KT& K,
    const bool normalise) {
  aux::matrix ws(1,w.size());
  row(ws,0) = w;
  return row(selfTreeDensity(tree, ws, N, K, normalise), 0);
}

template <class NT, class KT, class PT>
indii::ml::aux::vector indii::ml::aux::distributedSelfTreeDensity(
    PT& tree, const vector& w, const NT& N, const KT& K,
    const bool normalise) {
  aux::matrix ws(1,w.size());
  row(ws,0) = w;
  return row(distributedSelfTreeDensity(tree, ws, N, K, normalise), 0);
}

template <class NT, class KT, class PT>
indii::ml::aux::matrix indii::ml::aux::dualTreeDensity(
    PT& queryTree, PT& targetTree, const matrix& ws,
    const NT& N, const KT& K, const bool normalise) {
  /* pre-condition */
  assert (ws.size2() == targetTree.getData()->getSize());
  assert (queryTree.getData()->getDimensions() ==
      targetTree.getData()->getDimensions());
  
  DiracMixturePdf& q = *queryTree.getData();
  DiracMixturePdf& p = *targetTree.getData();
  PartitionTreeNode* queryRoot = queryTree.getRoot();
  PartitionTreeNode* targetRoot = targetTree.getRoot();

  matrix result(ws.size1(), q.getSize());
  result.clear();
  
  if (queryRoot != NULL && targetRoot != NULL) {
    std::stack<PartitionTreeNode*> queryNodes, targetNodes;

    vector x(p.getDimensions());
    unsigned int i, j;
    double w, d;

    queryNodes.push(queryRoot);
    targetNodes.push(targetRoot);
    
    while (!queryNodes.empty()) {
      PartitionTreeNode& queryNode = *queryNodes.top();
      queryNodes.pop();
      PartitionTreeNode& targetNode = *targetNodes.top();
      targetNodes.pop();

      if (queryNode.isLeaf() && targetNode.isLeaf()) {
        i = queryNode.getIndex();
        j = targetNode.getIndex();
        noalias(x) = q.get(i) - p.get(j);
        d = K(N(x));
        noalias(column(result,i)) += d * column(ws,j);
      } else if (queryNode.isLeaf() && targetNode.isPrune()) {
        i = queryNode.getIndex();
        const std::vector<unsigned int>& js = targetNode.getIndices();
        for (j = 0; j < js.size(); j++) {
          noalias(x) = q.get(i) - p.get(js[j]);
          d = K(N(x));
          noalias(column(result,i)) += d * column(ws,js[j]);
        }
      } else if (queryNode.isPrune() && targetNode.isLeaf()) {
        const std::vector<unsigned int>& is = queryNode.getIndices();
        j = targetNode.getIndex();
        for (i = 0; i < is.size(); i++) {
          noalias(x) = q.get(is[i]) - p.get(j);
          d = K(N(x));
          noalias(column(result,is[i])) += d * column(ws,j);
        }
      } else if (queryNode.isPrune() && targetNode.isPrune()) {
        const std::vector<unsigned int>& is = queryNode.getIndices();
        const std::vector<unsigned int>& js = targetNode.getIndices();
        for (i = 0; i < is.size(); i++) {
          for (j = 0; j < js.size(); j++) {
            noalias(x) = q.get(is[i]) - p.get(js[j]);
            d = K(N(x));
            noalias(column(result,is[i])) += d * column(ws,js[j]);
          }
        }
      } else {
        /* should we recurse? */
        targetNode.difference(queryNode, x);
        if (K(N(x)) > 0.0) {
          if (queryNode.isInternal()) {
            if (targetNode.isInternal()) {
              /* split both query and target nodes */
              queryNodes.push(queryNode.getLeft());
              targetNodes.push(targetNode.getLeft());
          
              queryNodes.push(queryNode.getLeft());
              targetNodes.push(targetNode.getRight());

              queryNodes.push(queryNode.getRight());
              targetNodes.push(targetNode.getLeft());

              queryNodes.push(queryNode.getRight());
              targetNodes.push(targetNode.getRight());        
            } else {
              /* split query node only */
              queryNodes.push(queryNode.getLeft());
              targetNodes.push(&targetNode);
          
              queryNodes.push(queryNode.getRight());
              targetNodes.push(&targetNode);
            }
          } else {
            /* split target node only */
            queryNodes.push(&queryNode);
            targetNodes.push(targetNode.getLeft());
        
            queryNodes.push(&queryNode);
            targetNodes.push(targetNode.getRight());
          }
        }
      }
    }
    
    if (normalise) {
      result /= p.getTotalWeight();
    }
  }
  
  return result;
}

template <class NT, class KT, class PT>
indii::ml::aux::matrix indii::ml::aux::distributedDualTreeDensity(
    PT& queryTree, PT& targetTree, const matrix& ws,
    const NT& N, const KT& K, const bool normalise) {
  boost::mpi::communicator world;
  const unsigned int size = world.size();
  unsigned int i;
  
  matrix result(dualTreeDensity(queryTree, targetTree, ws, N, K, false));
  rotate(*queryTree.getData());
  rotate(queryTree);
  rotate(result);

  for (i = 1; i < size; i++) {
    noalias(result) += dualTreeDensity(queryTree, targetTree, ws, N, K,
        false);
    rotate(*queryTree.getData());
    rotate(queryTree);
    rotate(result);
  }

  if (normalise) {
    result /= targetTree.getData()->getDistributedTotalWeight();
  }

  return result;
}

template <class NT, class KT, class PT>
indii::ml::aux::matrix indii::ml::aux::selfTreeDensity(PT& tree,
    const matrix& ws, const NT& N, const KT& K, const bool normalise) {
  /* pre-condition */
  assert (ws.size2() == tree.getData()->getSize());
  
  DiracMixturePdf& p = *tree.getData();
  PartitionTreeNode* root = tree.getRoot();

  matrix result(ws.size1(), p.getSize());
  result.clear();
  
  if (root != NULL) {
    std::stack<PartitionTreeNode*> queryNodes, targetNodes;
    std::stack<bool> doCrosses; // for query equals target tree optimisations

    vector x(p.getDimensions());
    unsigned int i, j;
    double w, d;
    bool doCross;

    queryNodes.push(root);
    targetNodes.push(root);
    doCrosses.push(false);
    
    while (!queryNodes.empty()) {
      PartitionTreeNode& queryNode = *queryNodes.top();
      queryNodes.pop();
      PartitionTreeNode& targetNode = *targetNodes.top();
      targetNodes.pop();
      doCross = doCrosses.top();
      doCrosses.pop();

      if (queryNode.isLeaf() && targetNode.isLeaf()) {
        i = queryNode.getIndex();
        j = targetNode.getIndex();
        noalias(x) = p.get(i) - p.get(j);
        d = K(N(x));
        if (doCross) {
          noalias(column(result,j)) += d * column(ws,i);
        }
        noalias(column(result,i)) += d * column(ws,j);
      } else if (queryNode.isLeaf() && targetNode.isPrune()) {
        i = queryNode.getIndex();
        const std::vector<unsigned int>& js = targetNode.getIndices();
        for (j = 0; j < js.size(); j++) {
          noalias(x) = p.get(i) - p.get(js[j]);
          d = K(N(x));
          if (doCross) {
            noalias(column(result,js[j])) += d * column(ws,i);
          }
          noalias(column(result,i)) += d * column(ws,js[j]);
        }
      } else if (queryNode.isPrune() && targetNode.isLeaf()) {
        const std::vector<unsigned int>& is = queryNode.getIndices();
        j = targetNode.getIndex();
        for (i = 0; i < is.size(); i++) {
          noalias(x) = p.get(is[i]) - p.get(j);
          d = K(N(x));
          if (doCross) {
            noalias(column(result,j)) += d * column(ws,is[i]);
          }
          noalias(column(result,is[i])) += d * column(ws,j);
        }
      } else if (queryNode.isPrune() && targetNode.isPrune()) {
        const std::vector<unsigned int>& is = queryNode.getIndices();
        const std::vector<unsigned int>& js = targetNode.getIndices();
        for (i = 0; i < is.size(); i++) {
          for (j = 0; j < js.size(); j++) {
            noalias(x) = p.get(is[i]) - p.get(js[j]);
            d = K(N(x));
            if (doCross) {
              noalias(column(result,js[j])) += d * column(ws,is[i]);
            }
            noalias(column(result,is[i])) += d * column(ws,js[j]);
          }
        }
      } else {
        /* should we recurse? */
        targetNode.difference(queryNode, x);
        if (K(N(x)) > 0.0) {
          if (queryNode.isInternal()) {
            if (targetNode.isInternal()) {
              /* split both query and target nodes */
              queryNodes.push(queryNode.getLeft());
              targetNodes.push(targetNode.getLeft());
              doCrosses.push(doCross);
          
              queryNodes.push(queryNode.getLeft());
              targetNodes.push(targetNode.getRight());
              if (&queryNode == &targetNode) {
                /* symmetric, so just double left-right evaluation */
                doCrosses.push(true);
              } else {
                /* asymmetric, so evaluate right-left separately */
                doCrosses.push(doCross);

                queryNodes.push(queryNode.getRight());
                targetNodes.push(targetNode.getLeft());
                doCrosses.push(doCross);
              }
              
              queryNodes.push(queryNode.getRight());
              targetNodes.push(targetNode.getRight());        
              doCrosses.push(doCross);
            } else {
              /* split query node only */
              queryNodes.push(queryNode.getLeft());
              targetNodes.push(&targetNode);
              doCrosses.push(doCross);
          
              queryNodes.push(queryNode.getRight());
              targetNodes.push(&targetNode);
              doCrosses.push(doCross);
            }
          } else {
            /* split target node only */
            queryNodes.push(&queryNode);
            targetNodes.push(targetNode.getLeft());
            doCrosses.push(doCross);
        
            queryNodes.push(&queryNode);
            targetNodes.push(targetNode.getRight());
            doCrosses.push(doCross);
          }
        }
      }
    }
    
    if (normalise) {
      result /= p.getTotalWeight();
    }
  }
  
  return result;
}

template <class NT, class KT, class PT>
indii::ml::aux::matrix indii::ml::aux::distributedSelfTreeDensity(
    PT& tree, const matrix& ws, const NT& N, const KT& K,
    const bool normalise) {
  boost::mpi::communicator world;
  const unsigned int size = world.size();
  
  matrix result(selfTreeDensity(tree, ws, N, K, false));
  
  if (size > 1) {
    /* cross densities */
    unsigned int crosses = (size - 1) / 2;
    bool leftover = (size - 1) % 2 > 0;
    unsigned int i;
  
    matrix ws2(ws);
    matrix result2(result.size1(), result.size2());
    result2.clear();
  
    PT* tree2 = dynamic_cast<PT*>(tree.clone());
    DiracMixturePdf q(*tree.getData());
    tree2->setData(&q);
    
    for (i = 0; i < crosses; i++) {
      rotate(*tree2->getData());
      rotate(*tree2);
      rotate(result2);
      rotate(ws2);
      crossTreeDensity(tree, *tree2, ws, ws2, N, K, result, result2,
          false, false);
    }

    if (leftover) {
      rotate(*tree2->getData());
      rotate(*tree2);
      rotate(ws2);
      noalias(result) += dualTreeDensity(tree, *tree2, ws2, N, K, false);
    }

    /* return results to original node */
    rotate(result2, size - i);
    noalias(result) += result2;

    delete tree2;
  }

  if (normalise) {
    result /= tree.getData()->getDistributedTotalWeight();
  }

  return result;
}

template <class NT, class KT, class PT>
void indii::ml::aux::crossTreeDensity(
    PT& tree1, PT& tree2, const matrix& ws1,
    const matrix& ws2, const NT& N, const KT& K, matrix& result1,
    matrix& result2, const bool clear, const bool normalise) {
  /* pre-condition */
  assert (ws1.size2() == tree1.getData()->getSize());
  assert (ws2.size2() == tree2.getData()->getSize());
  assert (result1.size2() == tree1.getData()->getSize());
  assert (result2.size2() == tree2.getData()->getSize());
  assert (result1.size1() == ws1.size1());
  assert (result2.size1() == ws2.size1());
  assert (tree1.getData()->getDimensions() ==
      tree2.getData()->getDimensions());
  
  DiracMixturePdf& p1 = *tree1.getData();
  DiracMixturePdf& p2 = *tree2.getData();
  PartitionTreeNode* root1 = tree1.getRoot();
  PartitionTreeNode* root2 = tree2.getRoot();

  if (clear) {
    result1.clear();
    result2.clear();
  }
  
  if (root1 != NULL && root2 != NULL) {
    std::stack<PartitionTreeNode*> nodes1, nodes2;

    vector x(p1.getDimensions());
    unsigned int i, j;
    double w, d;

    nodes1.push(root1);
    nodes2.push(root2);
    
    while (!nodes1.empty()) {
      PartitionTreeNode& node1 = *nodes1.top();
      nodes1.pop();
      PartitionTreeNode& node2 = *nodes2.top();
      nodes2.pop();

      if (node1.isLeaf() && node2.isLeaf()) {
        i = node1.getIndex();
        j = node2.getIndex();
        noalias(x) = p1.get(i) - p2.get(j);
        d = K(N(x));
        noalias(column(result1,i)) += d * column(ws2,j);
        noalias(column(result2,j)) += d * column(ws1,i);
      } else if (node1.isLeaf() && node2.isPrune()) {
        i = node1.getIndex();
        const std::vector<unsigned int>& js = node2.getIndices();
        for (j = 0; j < js.size(); j++) {
          noalias(x) = p1.get(i) - p2.get(js[j]);
          d = K(N(x));
          noalias(column(result1,i)) += d * column(ws2,js[j]);
          noalias(column(result2,js[j])) += d * column(ws1,i);
        }
      } else if (node1.isPrune() && node2.isLeaf()) {
        const std::vector<unsigned int>& is = node1.getIndices();
        j = node2.getIndex();
        for (i = 0; i < is.size(); i++) {
          noalias(x) = p1.get(is[i]) - p2.get(j);
          d = K(N(x));
          noalias(column(result1,is[i])) += d * column(ws2,j);
          noalias(column(result2,j)) += d * column(ws1,is[i]);
        }
      } else if (node1.isPrune() && node2.isPrune()) {
        const std::vector<unsigned int>& is = node1.getIndices();
        const std::vector<unsigned int>& js = node2.getIndices();
        for (i = 0; i < is.size(); i++) {
          for (j = 0; j < js.size(); j++) {
            noalias(x) = p1.get(is[i]) - p2.get(js[j]);
            d = K(N(x));
            noalias(column(result1,is[i])) += d * column(ws2,js[j]);
            noalias(column(result2,js[j])) += d * column(ws1,is[i]);
          }
        }
      } else {
        /* should we recurse? */
        node2.difference(node1, x);
        if (K(N(x)) > 0.0) {
          if (node1.isInternal()) {
            if (node2.isInternal()) {
              /* split both query and target nodes */
              nodes1.push(node1.getLeft());
              nodes2.push(node2.getLeft());
          
              nodes1.push(node1.getLeft());
              nodes2.push(node2.getRight());

              nodes1.push(node1.getRight());
              nodes2.push(node2.getLeft());
              
              nodes1.push(node1.getRight());
              nodes2.push(node2.getRight());        
            } else {
              /* split query node only */
              nodes1.push(node1.getLeft());
              nodes2.push(&node2);
          
              nodes1.push(node1.getRight());
              nodes2.push(&node2);
            }
          } else {
            /* split target node only */
            nodes1.push(&node1);
            nodes2.push(node2.getLeft());
        
            nodes1.push(&node1);
            nodes2.push(node2.getRight());
          }
        }
      }
    }
    
    if (normalise) {
      result1 /= p1.getTotalWeight();
      result2 /= p2.getTotalWeight();
    }
  }
}

#endif

