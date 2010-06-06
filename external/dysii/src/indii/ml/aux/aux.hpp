#ifndef INDII_ML_AUX_AUX_HPP
#define INDII_ML_AUX_AUX_HPP

/**
 * @file aux.hpp
 *
 * Precompilable header file. Includes all headers from
 * indii::ml::aux, which are commonly used by other sources. These
 * headers tend to make extensive use of templates and external
 * libraries, such that precompiling can lead to significant
 * performance gains.
 */

#include "vector.hpp"
#include "matrix.hpp"
#include "kde.hpp"
#include "parallel.hpp"

#include "Random.hpp"

#include "Pdf.hpp"
#include "GaussianPdf.hpp"
#include "DiracPdf.hpp"
#include "MixturePdf.hpp"
#include "GaussianMixturePdf.hpp"
#include "DiracMixturePdf.hpp"

#include "StochasticProcess.hpp"
#include "WienerProcess.hpp"

#endif
