//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "Pdf.hpp"
//#endif

using namespace indii::ml::aux;

Pdf::Pdf() : N(0) {
  //
}

Pdf::Pdf(const unsigned int N) : N(N) {
  //
}

Pdf::~Pdf() {
  //
}
