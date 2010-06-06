//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "FunctionCollection.hpp"

#include <assert.h>
#include <stdlib.h>

using namespace indii::ml::ode;

FunctionCollection::FunctionCollection() {
  //
}

FunctionCollection::FunctionCollection(const unsigned int N) : N(N), fs(N),
  fms(N), fvs(N), statics(N) {
  unsigned int i;
  for (i = 0; i < N; i++) {
    fs[i] = NULL;
    fms[i] = NULL;
    statics[i] = false;
  }
  this->object = this;
}

FunctionCollection::~FunctionCollection() {
  //
}

void FunctionCollection::setFunction(const unsigned int index, f_t value) {
  /* pre-condition */
  assert (index < N);

  fs[index] = value;
  fms[index] = NULL;
  statics[index] = true;
}

void FunctionCollection::setFunction(const unsigned int index,
    FunctionModel* value) {
  /* pre-condition */
  assert (index < N);

  fs[index] = NULL;
  fms[index] = value;
  statics[index] = false;
}

bool FunctionCollection::isStatic(const unsigned int index) {
  /* pre-condition */
  assert (index < N);

  return statics[index];
}

void FunctionCollection::evaluateAll(const double t, const double y[]) {
  unsigned int i;
  for (i = 0; i < N; i++) {
    if (statics[i]) {
      fvs[i] = fs[i](t, y, object);
    } else {
      fvs[i] = fms[i]->evaluate(t, y);
    }
  }
}

void FunctionCollection::setObject(void* object) {
  this->object = object;
}
