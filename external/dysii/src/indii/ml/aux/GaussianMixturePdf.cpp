//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "aux.hpp"
//#else
  #include "GaussianMixturePdf.hpp"
//#endif

using namespace indii::ml::aux;

GaussianMixturePdf::GaussianMixturePdf() : StandardMixturePdf<GaussianPdf>() {
  //
}

GaussianMixturePdf::GaussianMixturePdf(const unsigned int N) :
    StandardMixturePdf<GaussianPdf>(N) {
  //
}

GaussianMixturePdf::GaussianMixturePdf(const unsigned int K,
    const DiracMixturePdf& p) :
    StandardMixturePdf<GaussianPdf>(p.getDimensions()) {
  std::vector<unsigned int> clusters(p.getSize());
  unsigned int i, k;
  
  vector ws(K);
  std::vector<vector> mus;
  std::vector<symmetric_matrix> sigmas;
  std::vector<GaussianPdf> xs;

  /* initialisation */
  vector mu(getDimensions());
  symmetric_matrix sigma(getDimensions());
  GaussianPdf x(mu, sigma);  
  for (k = 0; k < K; k++) {
    mus.push_back(mu);
    sigmas.push_back(sigma);
    xs.push_back(x);
  }
  for (i = 0; i < p.getSize(); i++) {
    clusters[i] = i % K;
  }
  
  bool change = false;
  unsigned int k_max;
  double d, d_max;
  do {
    /* expectation */
    ws.clear();
    for (k = 0; k < K; k++) {
      mus[k].clear();
      sigmas[k].clear();
    }

    for (i = 0; i < p.getSize(); i++) {
      k = clusters[i];
      ws(k) += p.getWeight(i);
      noalias(mus[k]) += p.getWeight(i) * p.get(i);
      noalias(sigmas[k]) += p.getWeight(i) * outer_prod(p.get(i), p.get(i));
    }
    for (k = 0; k < K; k++) {
      mus[k] /= ws(k);
      sigmas[k] /= ws(k);
      sigmas[k] -= outer_prod(mus[k],mus[k]);
    }

    /* maximisation */
    for (k = 0; k < K; k++) {
      xs[k].setExpectation(mus[k]);
      xs[k].setCovariance(sigmas[k]);
    }
    
    change = false;
    for (i = 0; i < p.getSize(); i++) {
      k_max = K;
      d_max = 0.0;
      for (k = 0; k < K; k++) {
        d = ws(k) * xs[k].densityAt(p.get(i));
        if (d > d_max) {
          k_max = k;
          d_max = d;
        }
      }
      change = change || clusters[i] != k_max;
      clusters[i] = k_max;
    }
  } while (change);
  
  /* construct */
  for (k = 0; k < K; k++) {
    add(xs[k], ws(k));
  }
}

GaussianMixturePdf::~GaussianMixturePdf() {
  //
}

