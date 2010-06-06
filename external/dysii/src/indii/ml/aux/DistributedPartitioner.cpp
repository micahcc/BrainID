#include "DistributedPartitioner.hpp"

#include "PartitionFunctor.hpp"

#include <limits>
#include <vector>
#include <algorithm>

using namespace indii::ml::aux;

DistributedPartitioner::DistributedPartitioner(const unsigned int nth) :
    nth(nth) {
  //
}

DistributedPartitioner::~DistributedPartitioner() {
  //
}

bool DistributedPartitioner::init(DiracMixturePdf* p,
    const std::vector<unsigned int>& is) {
  boost::mpi::communicator world;
  const unsigned int rank = world.rank();
  const unsigned int size = world.size();

  const unsigned int N = p->getDimensions();
  unsigned int i, j;
  vector lower(N);
  vector upper(N);
  vector length(N);
  
  /* calculate local bounds */
  if (is.size() > 0) {
    noalias(lower) = p->get(is[0]);
    noalias(upper) = lower;
    
    for (i = 1; i < is.size(); i++) {
      for (j = 0; j < N; j++) {
        vector& x = p->get(is[i]);
        if (x(j) < lower(j)) {
          lower(j) = x(j);
        } else if (x(j) > upper(j)) {
          upper(j) = x(j);
        }
      }
    }  
  } else {
    noalias(lower) = scalar_vector(N,
        std::numeric_limits<double>::quiet_NaN());
    noalias(upper) = lower;
  }
 
  /* calculate global bounds */
  for (i = 0; i < N; i++) {
    lower(i) = all_reduce(world, lower(i), boost::mpi::minimum<double>());
    upper(i) = all_reduce(world, upper(i), boost::mpi::maximum<double>());
  }
  
  /* select longest dimension */
  noalias(length) = upper - lower;
  this->index = index_norm_inf(length);   

  /* split on nth element of this dimension */
  std::vector<unsigned int> js(is); // because of const-ness of argument
  std::vector<unsigned int>::iterator split, guess, from, to;
  unsigned int guesser = 0; // rank of node that guesses nth element
  double value, lastValue;
  unsigned int leftP, nth = this->nth;
  
  from = js.begin();
  to = js.end();  
  guess = from;

  do {
    /* guess of nth component */
    do {
      if (rank == guesser) {
        if (guess != to) {
          /* guess */
          value = p->get(*guess)(this->index);
        } else {
          /* pass on responsibility */
          value = std::numeric_limits<double>::quiet_NaN();
        }
      }
      boost::mpi::broadcast(world, value, guesser);

      if (isnan(value)) {
        guesser++;
      }
    } while (isnan(value) && guesser < size);
    
    /* if we've managed to find a value... */
    if (guesser < size) {
      lastValue = value;
      PartitionFunctor functor(*p, index, value);
  
      /* partition */
      split = std::stable_partition(from, to, functor);
      
      /* determine no. components either side */
      leftP = boost::mpi::all_reduce(world, std::distance(from, split),
          std::plus<unsigned int>());
      
      /* decide what to do next */
      if (leftP == 0) {
        /* no change, try a new guess */
        if (rank == guesser) {
          guess++;
        }
      } else if (leftP > nth) {
        /* nth is in left partition */
        to = split;
        guess = from;
      } else if (leftP < nth) {
        /* nth is in right partition */
        from = split;
        guess = from;
        nth -= leftP;
      }
    }
  } while (guesser < size && leftP != nth);
  
  this->value = lastValue;

  return length(this->index) > 0.0;
}

