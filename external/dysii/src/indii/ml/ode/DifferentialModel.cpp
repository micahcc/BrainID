//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "DifferentialModel.hpp"

using namespace indii::ml::ode;

DifferentialModel::DifferentialModel() {
  //
}

DifferentialModel::DifferentialModel(const unsigned int dimensions) :
    dimensions(dimensions) {
  //
}

DifferentialModel::~DifferentialModel() {
  //
}

unsigned int DifferentialModel::getDimensions() {
  return this->dimensions;
}
