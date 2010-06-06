#ifndef INDII_ML_AUX_PARALLEL_HPP
#define INDII_ML_AUX_PARALLEL_HPP

#include "vector.hpp"
#include "matrix.hpp"

#include "boost/mpi.hpp"

#include <functional>

namespace boost {
  namespace mpi {

    /* Omit these from documentation */
    /// @cond COMMUTATIVE

    template<>
    struct is_commutative<std::plus<unsigned int>, unsigned int> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<double>, double> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::vector>,
        indii::ml::aux::vector> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::matrix>,
        indii::ml::aux::matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::symmetric_matrix>,
        indii::ml::aux::symmetric_matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::lower_triangular_matrix>,
        indii::ml::aux::lower_triangular_matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::upper_triangular_matrix>,
        indii::ml::aux::upper_triangular_matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::identity_matrix>,
        indii::ml::aux::identity_matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::zero_matrix>,
        indii::ml::aux::zero_matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::scalar_matrix>,
        indii::ml::aux::scalar_matrix> : mpl::true_ {
      //
    };

    template<>
    struct is_commutative<std::plus<indii::ml::aux::sparse_matrix>,
        indii::ml::aux::sparse_matrix> : mpl::true_ {
      //
    };

    /// @endcond

  }
}

namespace indii {
  namespace ml {
    namespace aux {

  /**
   * Determine a node's share of some number of items.
   *
   * @param P The number of items.
   *
   * @return The node's share of this number of items. This is simply
   * @p P divided by the number of nodes, with lower ranks taking on
   * an additional item if there are remainders.
   */
  unsigned int shareOf(const unsigned int P);

  /**
   * Rotate variable values between all nodes. For \f$N\f$ nodes, node
   * \f$i\f$ sends its value to node \f$i+N-1\pmod{N}\f$ and receives
   * the value of node \f$i+1\pmod{N}\f$. This facilitates
   * calculations involving data stored across multiple nodes without
   * gathering all the data onto one node.
   *
   * @param x Value to rotate. Contains value received on return.
   * @param num Number of rotations to make. Specifying a number here
   * is more efficient than multiple calls.
   */
  template <class T>
  void rotate(T& x, const unsigned int num = 1);

    }
  }
}

inline unsigned int indii::ml::aux::shareOf(const unsigned int P) {
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();
  unsigned int P_local = P / size;
  if (rank < P % size) {
    P_local++;
  }

  return P_local;
}

template <class T>
void indii::ml::aux::rotate(T& x, const unsigned int num = 1) {
  boost::mpi::communicator world;
  boost::mpi::request reqSend, reqRecv;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();
  
  if (size > 1 && num > 0) {
    /*
     * Is it necessary to copy the sent object first in case it is 
     * overwritten? Presumably isend() serializes it into a buffer before
     * returning so we can blat it with irecv() without problems.
     */
    //T send(x);
    reqSend = world.isend((rank+size-num) % size, 0, x);
    reqRecv = world.irecv((rank+num) % size, 0, x);
    reqRecv.wait();
    reqSend.wait();
  }
}

#endif
