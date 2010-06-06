//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "ParameterCollection.hpp"

#include <assert.h>

using namespace indii::ml::ode;

ParameterCollection::ParameterCollection() {
  //
}

ParameterCollection::ParameterCollection(const unsigned int N) : N(N), ps(N) {
  //
}

ParameterCollection::~ParameterCollection() {
  //
}

void ParameterCollection::setParameter(const unsigned int index,
    const double value) {
  /* pre-condition */
  assert (index < N);

  ps[index] = value;
}

