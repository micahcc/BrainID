#ifndef INDII_ML_AUX_MIXTUREPDF_HPP
#define INDII_ML_AUX_MIXTUREPDF_HPP

#include "Pdf.hpp"

#include <vector>

namespace indii {
  namespace ml {
    namespace aux {
    
/**
 * Mixture probability density.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 571 $
 * @date $Date: 2008-09-24 15:06:22 +0100 (Wed, 24 Sep 2008) $
 *
 * @param T Component type, should be derivative of Pdf.
 *
 * @section MixturePdf_serialization Serialization
 *
 * This class supports serialization through the Boost.Serialization library.
 *
 * @section MixturePdf_parallel Parallelisation
 *
 * This class supports distributed storage. MixturePdf objects may be
 * instantiated on all nodes of a communicator, each with a different
 * set of mixture components, such that the union of all components
 * defines the full distribution.
 *
 * "Regular" methods such as getSize(), getTotalWeight() and
 * getExpectation() use only those components stored on the local
 * node. Special "distributed" methods such as
 * getDistributedSize(), getDistributedTotalWeight() and
 * getDistributedExpectation() use all components stored across the
 * nodes.
 */
template <class T>
class MixturePdf : public Pdf {
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
  MixturePdf();

  /**
   * Constructor. One or more components should be added with
   * add() after construction.
   *
   * @param N Dimensionality of the distribution.
   */
  MixturePdf(const unsigned int N);

  /**
   * Destructor.
   */
  virtual ~MixturePdf();

  /**
   * Assignment operator. Both sides must have the same dimensionality,
   * but may have different number of components.
   */
  MixturePdf<T>& operator=(const MixturePdf<T>& o);

  virtual void setDimensions(const unsigned int N,
      const bool preserve = false);

  /**
   * @name Local storage methods
   *
   * Use these methods if:
   *
   * @li You are not working in a parallel environment,
   * @li You are working in a parallel environment but are not using
   * the distributed storage features of this class, or
   * @li You are working in a parallel environment and are using the
   * distributed storage features of this class, but only want to deal
   * with the mixture components stored on the local node.
   */
  //@{

  /**
   * Add a component to the distribution on the local node.
   *
   * @param x The component.
   * @param w Unnormalised weight of the component.
   *
   * The new component is added to the end of the list of components in
   * terms of indices used by get(), getWeight(), etc.
   */
  void add(const T& x, const double w = 1.0);

  /**
   * Get component on the local node.
   *
   * @param i Index of the component.
   *
   * @return The @p i th component.
   */
  const T& get(const unsigned int i) const;

  /**
   * Get component on the local node.
   *
   * @param i Index of the component.
   *
   * @return The @p i th component.
   *
   * @note Modifying any of the components in the mixture may have
   * unintended side effects, especially since Pdf classes rely heavily on
   * precalculations which may consequently become out of date. This method
   * is provided only for those situations where precalculations must be
   * made within the component objects themselves, such as within their
   * getExpectation() methods, such that a non-const context is required.
   * For all other situations, favour the const version of this method.
   */
  T& get(const unsigned int i);

  /**
   * Set component on the local node.
   *
   * @param i Index of the component.
   * @param x Value of the component.
   */
  void set(const unsigned int i, const T&);

  /**
   * Get components on the local node.
   *
   * @return \f$\{(\mathbf{x}^{(i)},w^{(i)})\}\f$; set of weighted
   * components defining the distribution, as a vector.
   */
  const std::vector<T>& getAll() const;

  /**
   * Get components on the local node.
   *
   * @return \f$\{(\mathbf{x}^{(i)},w^{(i)})\}\f$; set of weighted
   * components defining the distribution, as a vector.
   *
   * @note Modifying any of the components in the mixture may have
   * unintended side effects, especially since Pdf classes rely heavily on
   * precalculations which may consequently become out of date. This method
   * is provided only for those situations where precalculations must be
   * made within the component objects themselves, such as within their
   * getExpectation() methods, such that a non-const context is required.
   * For all other situations, favour the const version of this method.
   */
  std::vector<T>& getAll();

  /**
   * Get component weight on the local node.
   *
   * @param i Index of the component.
   *
   * @return Weight of the @p i th component.
   */
  double getWeight(const unsigned int i) const;

  /**
   * Set component weight on the local node.
   *
   * @param i Index of the component.
   * @param w Weight of the @p i th component.
   */
  void setWeight(const unsigned int i, const double w);

  /**
   * Get the weights of all components on the local node.
   *
   * @return Vector of the weights of all components.
   */
  const vector& getWeights() const;
  const std::vector<double>& getCumulativeWeights() const;
  /**
   * Set the weights of all components on the local node.
   *
   * @param ws Vector of the weights of all components.
   */
  void setWeights(const vector& ws);
  
  /**
   * Clear all components from the distribution on the local node.
   */
  void clear();

  /**
   * Get the number of components in the distribution on the local
   * node.
   *
   * @return \f$K\f$; the number of components in the distribution.
   */
  unsigned int getSize() const;

  /**
   * Get the total weight of components on the local node.
   *
   * @return \f$W\f$; the total weight of components.
   */
  double getTotalWeight() const;

  /**
   * Normalise weights on the local node to sum to 1.
   *
   * @warning If in a distributed storage situation this is probably
   * not what you want. Consider distributedNormalise()
   * instead. Normalising the weights only on the local node will skew
   * the weighting of mixture components across all nodes.
   */
  void normalise();

  /**
   * Sample from the distribution on the local node.
   *
   * @return A sample from the distribution.
   */
  virtual vector sample();

  /**
   * Calculate the density on the local node at a given point.
   *
   * \f[p(\mathbf{x}) = \frac{1}{W}\sum_{i=1}^{K}w^{(i)}p^{(i)}(\mathbf{x})\f]
   *
   * @param x \f$\mathbf{x}\f$; the point at which to calculate the
   * density.
   *
   * @return \f$p(\mathbf{x})\f$; the density at \f$\mathbf{x}\f$.
   */
  virtual double densityAt(const vector& x);

  /**
   * Get the expected value of the distribution on the local node.
   *
   * @return \f$\mathbf{\mu}\f$; expected value of the distribution.
   */
  virtual const vector& getExpectation();

  //@}

  /**
   * @name Distributed storage methods
   *
   * Use these methods if:
   *
   * @li You are working in a parallel environment and are using the
   * distributed storage features of this class, and want to deal with
   * the mixture components stored across all nodes.
   *
   * These methods are used for distributed storage of mixtures. They
   * require synchronization and communication between all nodes in
   * the communicator, such that if any of these methods is called on
   * one node, the same method should eventually, and preferably as
   * soon as possible, be called on all other nodes in the same
   * communicator.
   */
  //@{

  /**
   * Get the number of components in the distribution across all
   * nodes.
   *
   * @return \f$\sum_i K_i\f$; the number of components in the distribution.
   */
  unsigned int getDistributedSize() const;

  /**
   * Get the total weight of components across all nodes.
   *
   * @return \f$\sum_i W_i\f$; the total weight of components across
   * all nodes.
   */
  double getDistributedTotalWeight() const;

  /**
   * Normalise weights across all nodes to sum to 1.
   */
  void distributedNormalise();

  /**
   * Sample from the full distribution.
   *
   * @return A sample from the full distribution.
   */
  virtual vector distributedSample();

  /**
   * Draw multiple samples from the full distribution. This is
   * significantly more efficient than multiple calls to
   * distributedSample(), as it involves less message passing.
   *
   * @param P \f$P\f$; number of samples to draw.
   *
   * @return Vector of samples drawn on the local node. The number of
   * samples drawn across all nodes will total \f$P\f$.
   */
  virtual std::vector<vector> distributedSample(const unsigned int P);

  /**
   * Calculate the density of the full distribution at a given point.
   *
   * @param x \f$\mathbf{x}\f$; the point at which to calculate the
   * density.
   *
   * @return \f$p(\mathbf{x})\f$; the density at \f$\mathbf{x}\f$.
   */
  virtual double distributedDensityAt(const vector& x);

  /**
   * Perform multiple density calculations from the full distribution. This
   * is significantly more efficient than multiple calls to
   * distributedDensityAt(const vector&), as it involves less message
   * passing.
   *
   * @param xs The points on this node at which to calculate densities.
   * 
   * @return The densities at the given points on this node.
   *
   * Note that while each node is passed only its set of points and returns
   * only the density calculations for its set of points, all nodes
   * participate in the calculation for all points.
   *
   * @see distributedDensityAt(const vector&);
   */
  virtual vector distributedDensityAt(std::vector<vector>& xs);

  /**
   * Get the expected value of the full distribution.
   *
   * @return \f$\mathbf{\mu}\f$; expected value of the full
   * distribution.
   */
  virtual vector getDistributedExpectation();

  /**
   * Redistribute components across nodes by number. This attempts to
   * redistribute the components of the full distribution across
   * nodes, so that each node stores as close to an equal number of
   * components as possible.
   */
  void redistributeBySize();

  /**
   * Redistribute components across nodes by weight. This attempts to
   * redistribute the components of the full distribution across
   * nodes, so that the total weight of components at each node is as
   * close to an equal number as possible.
   */
  void redistributeByWeight();
  
  /**
   * Gathers all components to a single node
   * 
   *  @param node to gather to
   */
  void gatherToNode(unsigned int dst);

  //@}

  /**
   * Called when changes are made to the distribution, such as when a
   * new component is added. This causes pre-calculations to be
   * refreshed.
   */
  virtual void dirty();

private:
  /**
   * Node property.
   */
  template <class P>
  struct node_property {
    /**
     * Node rank.
     */
    unsigned int rank;

    /**
     * Property at node.
     */
    P prop;

    /**
     * Comparison operator for sorting.
     *
     * @return True if this node's property is greater than the
     * argument's property.
     */
    bool operator<(const node_property& o) const;

    /**
     * Serialize, or restore from serialization.
     */
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);

    /*
     * Boost.Serialization requirements.
     */
    friend class boost::serialization::access;

  };

  typedef struct node_property<int> node_count;
  typedef std::vector<node_count> node_count_vector;
  typedef typename node_count_vector::iterator node_count_iterator;

  typedef struct node_property<double> node_weight;
  typedef std::vector<node_weight> node_weight_vector;
  typedef typename node_weight_vector::iterator node_weight_iterator;

  /**
   * Components.
   */
  std::vector<T> xs;
  
  /**
   * Component weights.
   */
  indii::ml::aux::vector ws;
  
  /**
   * Cumulative component weights.
   */
  std::vector<double> Ws;

  /**
   * Last calculated expectation.
   */
  vector mu;

  /**
   * Last calculated unnormalised expectation.
   */
  vector Zmu;

  /**
   * Is the last calculated expectation up to date?
   */
  bool haveMu;

  /**
   * Calculate expectation from current components on the local node.
   */
  void calculateExpectation();

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

#include "Random.hpp"
#include "parallel.hpp"

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/vector.hpp"

#include "boost/mpi.hpp"
#include "boost/mpi/environment.hpp"
#include "boost/mpi/communicator.hpp"

#include <algorithm>

template <class T>
indii::ml::aux::MixturePdf<T>::MixturePdf() : xs(0), ws(0), Ws(0),
    mu(0), Zmu(0) {
  haveMu = false;
}

template <class T>
indii::ml::aux::MixturePdf<T>::MixturePdf(const unsigned int N) : Pdf(N),
    xs(0), ws(0), Ws(0), mu(N), Zmu(N) {
  haveMu = false;
}

template <class T>
indii::ml::aux::MixturePdf<T>::~MixturePdf() {
  //
}

template <class T>
indii::ml::aux::MixturePdf<T>& indii::ml::aux::MixturePdf<T>::operator=(
    const MixturePdf<T>& o) {
  /* pre-condition */
  assert (N == o.N);

  xs = o.xs;

  ws.resize(o.ws.size(), false);
  ws = o.ws;

  Ws.resize(o.Ws.size(), false);
  Ws = o.Ws;

  haveMu = o.haveMu;
  if (haveMu) {
    mu = o.mu;
    Zmu = o.Zmu;
  }
  
  return *this;
}

template <class T>
void indii::ml::aux::MixturePdf<T>::add(const T& x, const double w) {
  /* pre-condition */
  assert (x.getDimensions() == getDimensions());

  /* component */
  xs.push_back(x);
  
  /* weight */
  ws.resize(ws.size() + 1, true);
  ws(ws.size() - 1) = w;

  /* cumulative weight */
  Ws.push_back(getTotalWeight() + w);
    
  dirty();
  
  /* post-condition */
  assert (xs.size() == ws.size());
  assert (xs.size() == Ws.size());
}

template <class T>
inline const T& indii::ml::aux::MixturePdf<T>::get(const unsigned int i)
    const {
  return xs[i];
}

template <class T>
inline T& indii::ml::aux::MixturePdf<T>::get(const unsigned int i) {
  return xs[i];
}

template <class T>
void indii::ml::aux::MixturePdf<T>::set(const unsigned int i, const T& x) {
  xs[i] = x;
  dirty();
}

template <class T>
inline const std::vector<T>& indii::ml::aux::MixturePdf<T>::getAll() const {
  return xs;
}

template <class T>
inline std::vector<T>& indii::ml::aux::MixturePdf<T>::getAll() {
  return xs;
}

template <class T>
inline double indii::ml::aux::MixturePdf<T>::getWeight(
    const unsigned int i) const {
  return ws(i);
}

template <class T>
void indii::ml::aux::MixturePdf<T>::setWeight(const unsigned int i,
    const double w) {
  /* pre-condition */
  assert (i < getSize());
    
  this->ws(i) = w;

  /* recalculate cumulative weights */
  unsigned int j = i;
  if (j == 0) {
    Ws[0] = ws(0);
    j++;
  }
  for (; j < getSize(); j++) {
    Ws[j] = Ws[j - 1] + ws(j);
  }
  
  dirty();
}

template <class T>
inline const indii::ml::aux::vector&
    indii::ml::aux::MixturePdf<T>::getWeights() const {
  return ws;
}

template <class T>
inline const std::vector<double>&
    indii::ml::aux::MixturePdf<T>::getCumulativeWeights() const {
  return Ws;
}

template <class T>
void indii::ml::aux::MixturePdf<T>::setWeights(const vector& ws) {
  /* pre-condition */
  assert (this->ws.size() == ws.size());

  this->ws = ws;

  /* recalculate cumulative weights */
  unsigned int i = 0;
  if (getSize() > 0) {
    Ws[0] = ws(0);
    i++;
  }
  for (; i < getSize(); i++) {
    Ws[i] = Ws[i - 1] + ws(i);
  }
  
  dirty();
}

template <class T>
inline unsigned int indii::ml::aux::MixturePdf<T>::getSize() const {
  return xs.size();
}

template <class T>
void indii::ml::aux::MixturePdf<T>::normalise() {
  if (!Ws.empty()) {
    const double W = getTotalWeight();
  
    if (W != 1.0) {
      double WI = 1.0 / W;
    
      /* update weights */
      ws *= WI;
    
      /* update cumulative weights */
      std::vector<double>::iterator iter, end;
      iter = Ws.begin();
      end = Ws.end();
      while (iter != end) {
        *iter *= WI;
        iter++;
      }
      
      dirty(); // unnormalised mean out of date
    }
  }
}

template <class T>
void indii::ml::aux::MixturePdf<T>::clear() {
  xs.clear();
  ws.resize(0, false);
  Ws.clear();

  dirty();
}

template <class T>
double indii::ml::aux::MixturePdf<T>::getTotalWeight() const {
  return (Ws.empty() ? 0.0 : Ws.back());
}

template <class T>
void indii::ml::aux::MixturePdf<T>::setDimensions(const unsigned int N,
    const bool preserve) {
  this->N = N;

  mu.resize(N, preserve);
  Zmu.resize(N, preserve);

  if (preserve) {
    unsigned int i;
    for (i = 0; i < xs.size(); i++) {
      xs[i].setDimensions(N, preserve);
    }
  } else {
    clear();
  }

  dirty();
}

template <class T>
const indii::ml::aux::vector&
    indii::ml::aux::MixturePdf<T>::getExpectation() {
  if (!haveMu) {
    calculateExpectation();
  }
  return mu;
}

template <class T>
indii::ml::aux::vector indii::ml::aux::MixturePdf<T>::sample() {
  /* pre-condition */
  assert (!xs.empty());

  double u = Random::uniform(0.0, getTotalWeight());
  unsigned int i = std::distance(Ws.begin(), lower_bound(Ws.begin(),
      Ws.end(), u));
  
  return xs[i].sample();
}

template <class T>
double indii::ml::aux::MixturePdf<T>::densityAt(const vector& x) {
  double p = 0.0;
  if (getTotalWeight() > 0.0) {
    unsigned int i;
    for (i = 0; i < xs.size(); i++) {
      p += ws(i) * xs[i].densityAt(x);
    }
    p /= getTotalWeight();
  }

  /* post-condition */
  assert (p >= 0.0);

  return p;
}

template <class T>
unsigned int indii::ml::aux::MixturePdf<T>::getDistributedSize()
    const {
  boost::mpi::communicator world;

  return boost::mpi::all_reduce(world, getSize(), std::plus<unsigned int>());
}

template <class T>
double indii::ml::aux::MixturePdf<T>::getDistributedTotalWeight() const {
  boost::mpi::communicator world;
  return boost::mpi::all_reduce(world, getTotalWeight(),
      std::plus<double>()); 
}

template <class T>
void indii::ml::aux::MixturePdf<T>::distributedNormalise() {
  double W_d = getDistributedTotalWeight();

  if (W_d > 0.0 && W_d != 1.0) {
    double WI_d = 1.0 / W_d;

    /* update weights */
    ws *= WI_d;
  
    /* update cumulative weights */
    std::vector<double>::iterator iter, end;
    iter = Ws.begin();
    end = Ws.end();
    while (iter != end) {
      *iter *= WI_d;
      iter++;
    }

    dirty(); // unnormalised mean out of date
  }
}

template <class T>
indii::ml::aux::vector indii::ml::aux::MixturePdf<T>::distributedSample() {
  boost::mpi::communicator world;
  unsigned int rank = world.rank();
  unsigned int size = world.size();
  unsigned int i;

  double W_s;
  double u;
  vector x(N);

  /* scan sum weights across nodes */
  W_s = boost::mpi::scan(world, getTotalWeight(), std::plus<double>());

  /* final node has total weight, so it generatea random number up
     to that weight and broadcasts */
  if (rank == size - 1) {
    u = Random::uniform(0.0, W_s);
  }
  boost::mpi::broadcast(world, u, size - 1);

  if (u >= W_s - getTotalWeight() && u < W_s) {
    /* this node draws sample and sends */
    std::vector<boost::mpi::request> reqs;
    reqs.reserve(size - 1);
    x = sample();
    for (i = 0; i < size; i++) {
      if (i != rank) {
        reqs.push_back(world.isend(i, 0, x));
      }
    }

    // boost::mpi::wait_all seems exceptionally slow here, workaround:
    std::vector<boost::mpi::request>::iterator iter, end;
    iter = reqs.begin();
    end = reqs.end();
    while (iter != end) {
      iter->wait();
      iter++;
    }
    // end workaround

  } else {
    /* this node receives sample */
    world.recv(boost::mpi::any_source, boost::mpi::any_tag, x);
  }

  return x;
}

template <class T>
std::vector<indii::ml::aux::vector>
    indii::ml::aux::MixturePdf<T>::distributedSample(const unsigned int P) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  unsigned int i;

  double W_s;
  double u;
  std::vector<double> us;
  std::vector<double>::iterator iter, end;
  std::vector<vector> xs;

  /* scan sum weights across nodes */
  W_s = boost::mpi::scan(world, getTotalWeight(), std::plus<double>());

  /* final node has total weight, so it can generate random numbers up
     to total weight and broadcast to other nodes */
  if (rank == size - 1) {
    for (i = 0; i < P; i++) {
      us.push_back(Random::uniform(0.0, W_s));
    }
  }
  boost::mpi::broadcast(world, us, size - 1);

  iter = us.begin();
  end = us.end();
  while (iter != end) {
    u = *iter;
    if (u >= W_s - getTotalWeight() && u < W_s) {
      /* this node draws sample */
      xs.push_back(sample());
    }
    iter++;
  }

  return xs;
}

template <class T>
double indii::ml::aux::MixturePdf<T>::distributedDensityAt(const vector& x) {
  boost::mpi::communicator world;  
  double p = boost::mpi::all_reduce(world, densityAt(x), std::plus<double>());
  
  /* post-condition */
  assert (p >= 0.0);

  return p;
}

template <class T>
indii::ml::aux::vector indii::ml::aux::MixturePdf<T>::distributedDensityAt(
    std::vector<vector>& xs) {
  boost::mpi::communicator world;  
  unsigned int size = world.size();

  unsigned int i, k;
  vector p(xs.size());

  p.clear();
  for (k = 0; k < size; k++) {
    for (i = 0; i < xs.size(); i++) {
      p(i) += getTotalWeight() * densityAt(xs[i]);
    }
    rotate(p);
    rotate(xs);
  }
  p /= getDistributedTotalWeight();

  return p;
}

template <class T>
indii::ml::aux::vector
    indii::ml::aux::MixturePdf<T>::getDistributedExpectation() {
  boost::mpi::communicator world;
  const unsigned int size = world.size();
  
  if (size == 0) {
    return getExpectation();
  } else {
    vector mu_d(N);

    if (getTotalWeight() > 0.0) {
      if (!haveMu) {
        calculateExpectation();
      }
    } else {
      Zmu.clear();
    }

    noalias(mu_d) = boost::mpi::all_reduce(world, Zmu, std::plus<vector>());
    mu_d /= getDistributedTotalWeight();

    return mu_d;
  }
}

template <class T>
void indii::ml::aux::MixturePdf<T>::redistributeBySize() {
  namespace aux = indii::ml::aux;
  namespace ublas = boost::numeric::ublas;

  boost::mpi::communicator world;
  unsigned int rank = world.rank();
  unsigned int size = world.size();

  std::vector<T> xsBuffer;
  aux::vector wsBuffer;

  boost::mpi::request reqSendXs, reqSendWs;
  boost::mpi::request reqRecvXs, reqRecvWs;

  unsigned int P_total;
  unsigned int P_target;  // target no. components per node
  unsigned int P_change;  // no. components to transfer

  node_count excess, from, to;
  node_count_vector excesses;  // no. excess components at each node

  unsigned int i;

  /* work out target number of components for this node */
  P_total = getDistributedSize();
  P_target = P_total / size;
  if (rank < P_total % size) {
    P_target++; // take a leftover
  }

  /* gather excess components at each node */
  excess.rank = rank;
  excess.prop = getSize() - P_target;

  boost::mpi::all_gather(world, excess, excesses);
  std::stable_sort(excesses.begin(), excesses.end());

  /* while distribution of components is not even */
  while (excesses.front().prop > 0) {
    from = excesses.front();  // node with largest excess
    to = excesses.back();  // node with smallest (largest -ve) excess
    excesses.erase(excesses.begin());
    excesses.pop_back();

    P_change = std::min(abs(to.prop), from.prop);

    if (rank == from.rank) {
      /* send components */
      xsBuffer.clear();
      xsBuffer.insert(xsBuffer.end(), xs.end() - P_change, xs.end());
      xs.resize(xs.size() - P_change);
      reqSendXs = world.isend(to.rank, 0, xsBuffer);
      
      /* send weights */
      wsBuffer.resize(P_change, false);
      noalias(wsBuffer) = project(ws, ublas::range(ws.size() - P_change,
          ws.size()));
      ws.resize(ws.size() - P_change, true);
      reqSendWs = world.isend(to.rank, 1, wsBuffer);
      
      /* update cumulative weights */
      Ws.resize(Ws.size() - P_change);
      
      /* wait */
      reqSendWs.wait();
      reqSendXs.wait();
    } else if (rank == to.rank) {
      /* receive components */
      reqRecvXs = world.irecv(from.rank, 0, xsBuffer);
      reqRecvWs = world.irecv(from.rank, 1, wsBuffer);
      reqRecvWs.wait();
      reqRecvXs.wait();
      
      assert (xsBuffer.size() == wsBuffer.size());
      
      for (i = 0; i < xsBuffer.size(); i++) {
        add(xsBuffer[i], wsBuffer(i));
      }
    }

    /* update excesses */
    from.prop -= P_change;
    to.prop += P_change;
  
    /* re-sort */
    excesses.push_back(from); // 'from' must have >= no. components of 'to'
    excesses.push_back(to);
    inplace_merge(excesses.begin(), excesses.end() - 2, excesses.end());
  }

  dirty();
  
  /* post-conditions */
  assert (xs.size() == ws.size());
  assert (xs.size() == Ws.size());
}

template <class T>
void indii::ml::aux::MixturePdf<T>::redistributeByWeight() {
  boost::mpi::communicator world;
  unsigned int rank = world.rank();
  unsigned int size = world.size();

  std::vector<T> xsBuffer;
  aux::vector wsBuffer;

  boost::mpi::request reqSendXs, reqSendWs;
  boost::mpi::request reqRecvXs, reqRecvWs;

  double W_target = getDistributedTotalWeight() / size;
       // target weight per node
  double W_change;  // weight to transfer
  double W_actual;  // actual weight transferred

  node_weight excess, from, to;
  node_weight_vector excesses;  // excess weight at each node

  unsigned int i;

  /* gather excess weight at each node */
  excess.rank = rank;
  excess.prop = getTotalWeight() - W_target;
  boost::mpi::all_gather(world, excess, excesses);
  std::stable_sort(excesses.begin(), excesses.end());

  /* while distribution of components is not even */
  from = excesses.front();
  to = from;  // will never enter the loop if only one node
  while ((from.rank != excesses.front().rank ||
      to.rank != excesses.back().rank) &&
      excesses.front().prop != excesses.back().prop) {
    from = excesses.front();  // node with largest excess
    to = excesses.back();  // node with smallest (largest -ve) excess
    excesses.erase(excesses.begin());
    excesses.pop_back();

    W_change = std::min(fabs(to.prop), from.prop);

    if (rank == from.rank) {
      /* send components of up to W_change weight */
      W_actual = 0.0;
      xsBuffer.clear();
      wsBuffer.resize(0, false);
      
      while (W_actual + ws(ws.size() - 1) < W_change) {
        /* component */
        xsBuffer.push_back(xs.back());
        xs.pop_back();
        
        /* weight */
        wsBuffer.resize(wsBuffer.size() + 1, true);
        wsBuffer(wsBuffer.size() - 1) = ws(ws.size() - 1);
        ws.resize(ws.size() - 1, true);
        
        /* cumulative weights */
        Ws.pop_back();
        
        W_actual += wsBuffer(wsBuffer.size() - 1);
      }
      
      reqSendXs = world.isend(to.rank, 0, xsBuffer);
      reqSendWs = world.isend(to.rank, 1, wsBuffer);
      reqSendWs.wait();
      reqSendXs.wait();      

    } else if (rank == to.rank) {
      /* receive components */
      reqRecvXs = world.irecv(from.rank, 0, xsBuffer);
      reqRecvWs = world.irecv(from.rank, 1, wsBuffer);
      reqRecvWs.wait();
      reqRecvXs.wait();
      
      assert (xsBuffer.size() == wsBuffer.size());
      
      for (i = 0; i < xsBuffer.size(); i++) {
        add(xsBuffer[i], wsBuffer(i));
      }
    }

    /* broadcast actual weight transfer to all nodes */
    boost::mpi::broadcast(world, W_actual, from.rank);

    /* update excesses */
    from.prop -= W_actual;
    to.prop += W_actual;

    /* re-sort */
    excesses.push_back(from); // 'from' must have >= weight of 'to'
    excesses.push_back(to);
    inplace_merge(excesses.begin(), excesses.end() - 2, excesses.end());
  }

  dirty();

  /* post-conditions */
  assert (xs.size() == ws.size());
  assert (xs.size() == Ws.size());
}

/** 
 * gatherToNode 
 * 
 * @param dst - destination rank
 * todo: stop using add (would make much faster)
**/
template <class T>
void indii::ml::aux::MixturePdf<T>::gatherToNode(unsigned int dst)
{
  boost::mpi::communicator world;
  unsigned int rank = world.rank();
  unsigned int size = world.size();
  
  assert(dst < size);

  std::vector< std::vector< T > > xsFull;
  std::vector< aux::vector > wsFull;

  #ifndef NDEBUG
  unsigned int initialSize = input.getDistributedSize();
  aux::vector initialMu = input.getDistributedExpectation();
  aux::matrix initialCov = input.getDistributedCovariance();
  #endif 

  /* if rank is the destination then receive from all the other nodes */
  if(rank == dst) {
    /* Receive from each other node */
    boost::mpi::gather(world, getAll(), xsFull, dst); 
    boost::mpi::gather(world, getWeights(), wsFull, dst); 

    for(unsigned int ii=0 ; ii < size ; ii++) {
      if(ii != rank) {
        for (unsigned int jj = 0; jj < xsFull[ii].size(); jj++) {
          add( (xsFull[ii])[jj] , (wsFull[ii])(jj) );
        }
      }
    }
  
  /* if rank is not the destination then send to the destination */
  } else {
    boost::mpi::gather(world, getAll(), dst); 
    boost::mpi::gather(world, getWeights(), dst); 
    clear();
  }
  
  #ifndef NDEBUG
  unsigned int endSize = input.getDistributedSize();
  aux::vector endMu = input.getDistributedExpectation();
  aux::matrix endCov = input.getDistributedCovariance();
  assert (initialSize == endSize);
  #endif 
  dirty();
};


template <class T>
void indii::ml::aux::MixturePdf<T>::dirty() {
  haveMu = false;
}

template <class T>
void indii::ml::aux::MixturePdf<T>::calculateExpectation() {
  /* pre-condition */
  assert (getTotalWeight() > 0.0);

  unsigned int i;

  Zmu.clear();
  for (i = 0; i < xs.size(); i++) {
    noalias(Zmu) += ws(i) * xs[i].getExpectation();
  }
  noalias(mu) = Zmu / getTotalWeight();
  haveMu = true;
}

template <class T>
template <class Archive>
void indii::ml::aux::MixturePdf<T>::save(Archive& ar,
    const unsigned int version) const {
  ar & boost::serialization::base_object<Pdf>(*this);

  ar & xs;
  ar & ws;
  ar & Ws;
}

template <class T>
template <class Archive>
void indii::ml::aux::MixturePdf<T>::load(Archive& ar,
    const unsigned int version) {
  ar & boost::serialization::base_object<Pdf>(*this);

  ar & xs;
  ar & ws;
  ar & Ws;

  mu.resize(N, false);
  Zmu.resize(N, false);
  haveMu = false;
}

template <class T>
template <class P>
bool indii::ml::aux::MixturePdf<T>::node_property<P>::operator<(
    const node_property& o) const {
  return prop > o.prop;
}

template <class T>
template <class P>
template <class Archive>
void indii::ml::aux::MixturePdf<T>::node_property<P>::serialize(Archive& ar,
    const unsigned int version) {
  ar & rank;
  ar & prop;
}

#endif

