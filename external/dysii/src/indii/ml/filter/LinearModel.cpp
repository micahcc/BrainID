//#if defined(__GNUC__) && defined(GCC_PCH)
//  #include "../aux/aux.hpp"
//#endif

#include "LinearModel.hpp"

using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

/**
 * @todo System noise needn't have the same number of dimensions at the state,
 * and likewise G may be N*P, not N*N, where P is the number of dimensions of
 * the noise.
 */
LinearModel::LinearModel(aux::matrix& A, aux::matrix& G,
    aux::symmetric_matrix& Q, aux::matrix& C, aux::symmetric_matrix& R) :
    N(A.size1()), M(C.size1()), A(A), AT(trans(A)), G(G), GT(trans(G)), Q(Q),
    C(C), CT(trans(C)), R(R), AI(N,N) {
  /* pre-condition */
  #ifndef NDEBUG
  assert(A.size1() == N);
  assert(A.size2() == N);
  assert(G.size1() == N);
  assert(G.size2() == N);
  assert(Q.size1() == N);
  assert(Q.size2() == N);
  assert(C.size1() == M);
  assert(C.size2() == N);
  assert(R.size1() == M);
  assert(R.size2() == M);
  #endif

  /* for backwards pass */
  aux::matrix tmp(A);
  aux::inv(tmp,AI);
}

LinearModel::~LinearModel() {
  //
}

aux::GaussianPdf LinearModel::p_xtnp1_ytn(const aux::GaussianPdf& p_xtn_ytn,
      const unsigned int delta) {
  const aux::vector& x_xtn_ytn = p_xtn_ytn.getExpectation();
  const aux::symmetric_matrix& P_xtn_ytn = p_xtn_ytn.getCovariance();
  aux::vector x_xtnp1_ytn(N);
  aux::symmetric_matrix P_xtnp1_ytn(N,N);

  noalias(x_xtnp1_ytn) = prod(A,x_xtn_ytn);
  aux::matrix X(N,N);
  noalias(X) = prod(P_xtn_ytn,AT);
  noalias(P_xtnp1_ytn) = prod(A,X);
  noalias(X) = prod(Q,GT);
  noalias(P_xtnp1_ytn) += prod(G,X);

  return aux::GaussianPdf(x_xtnp1_ytn, P_xtnp1_ytn);
}

/**
 * @todo Doesn't consider delta currently. Should iterate as many
 * times as specified by delta, for example, in the case that there is
 * a missing observation. Just do this using recursion.
 */
aux::GaussianPdf LinearModel::p_xtnp1_ytnp1(
    const aux::GaussianPdf& p_xtnp1_ytn, const aux::vector& ytnp1,
    const unsigned int delta) {
  const aux::vector& x_xtnp1_ytn = p_xtnp1_ytn.getExpectation();
  const aux::symmetric_matrix& P_xtnp1_ytn = p_xtnp1_ytn.getCovariance();
  aux::vector x_xtnp1_ytnp1(N);
  aux::symmetric_matrix P_xtnp1_ytnp1(N);
  aux::matrix X(M,M), Y(M,M), Z(N,M), K_tnp1(N,M);

  noalias(Z) = prod(P_xtnp1_ytn, CT);
  noalias(X) = prod(C, Z) + R;
  aux::inv(X, Y);

  noalias(Z) = prod(P_xtnp1_ytn, CT);
  noalias(K_tnp1) = prod(Z, Y); // kalman gain

  noalias(x_xtnp1_ytnp1) = x_xtnp1_ytn
      + prod(K_tnp1,aux::vector(ytnp1 - prod(C,x_xtnp1_ytn)));
  noalias(P_xtnp1_ytnp1) = P_xtnp1_ytn
      - prod(K_tnp1,aux::matrix(prod(C,P_xtnp1_ytn)));

  return aux::GaussianPdf(x_xtnp1_ytnp1, P_xtnp1_ytnp1);
}

aux::GaussianPdf LinearModel::p_y_x(const aux::GaussianPdf& p_x) {
  const aux::vector& x = p_x.getExpectation();
  const aux::symmetric_matrix& P_x = p_x.getCovariance();
  aux::vector y(M);
  aux::symmetric_matrix P_y(M,M);
  aux::matrix X(M,M);

  noalias(y) = prod(C,x);

  noalias(X) = prod(P_x, CT);
  noalias(P_y) = prod(C, X) + R;

  return aux::GaussianPdf(y, P_y);
}

aux::GaussianPdf LinearModel::p_xtn_ytT(const aux::GaussianPdf& p_xtnp1_ytT,
    const aux::GaussianPdf& p_xtnp1_ytn, const aux::GaussianPdf& p_xtn_ytn,
    const unsigned int delta) {
  const aux::vector& x_xtnp1_ytT = p_xtnp1_ytT.getExpectation();
  const aux::symmetric_matrix& P_xtnp1_ytT = p_xtnp1_ytT.getCovariance();
  const aux::vector& x_xtnp1_ytn = p_xtnp1_ytn.getExpectation();
  const aux::symmetric_matrix& P_xtnp1_ytn = p_xtnp1_ytn.getCovariance();
  const aux::vector& x_xtn_ytn = p_xtn_ytn.getExpectation();
  const aux::symmetric_matrix& P_xtn_ytn = p_xtn_ytn.getCovariance();
  aux::vector x_xtn_ytT(N);
  aux::symmetric_matrix P_xtn_ytT(N);

  aux::matrix L_tn(N,N), LT_tn(N,N), PI_xtnp1_ytn(N,N), X(N,N);
  noalias(X) = P_xtnp1_ytn;
  aux::inv(X, PI_xtnp1_ytn);

  noalias(X) = prod(AT, PI_xtnp1_ytn);
  noalias(L_tn) = prod(P_xtn_ytn, X);
  noalias(LT_tn) = trans(L_tn);

  noalias(x_xtn_ytT) = x_xtn_ytn + prod(L_tn, x_xtnp1_ytT - x_xtnp1_ytn);
  noalias(X) = prod(P_xtnp1_ytT - P_xtnp1_ytn, LT_tn);
  noalias(P_xtn_ytT) = P_xtn_ytn + prod(L_tn, X);

  return aux::GaussianPdf(x_xtn_ytT, P_xtn_ytT);
}

aux::GaussianPdf LinearModel::p_xtnm1_ytn(const aux::GaussianPdf& p_xtn_ytn,
    const unsigned int delta) {
  /* calculate inverse dynamics */
  const aux::vector x_xtn_ytn = p_xtn_ytn.getExpectation();
  const aux::matrix P_xtn_ytn = p_xtn_ytn.getCovariance();

  aux::matrix A_b(N,N);
  aux::matrix G_b(N,N);
  aux::matrix Q_b(N,N);
  aux::vector w_b(N);

  aux::matrix PI_xtn_ytn(N,N);
  aux::identity_matrix I(N);
  aux::matrix X(N,N);

  noalias(X) = P_xtn_ytn;
  aux::inv(X, PI_xtn_ytn);

  noalias(A_b) = prod(AI,G);
  A_b = prod(A_b,Q);
  A_b = prod(A_b,GT);
  A_b = prod(A_b,PI_xtn_ytn);
  A_b = I - A_b;
  A_b = prod(AI, A_b);

  noalias(G_b) = -1.0 * prod(AI,G);

  noalias(Q_b) = prod(Q,GT);
  Q_b = prod(Q_b,PI_xtn_ytn);
  Q_b = prod(Q_b,G);
  Q_b = prod(Q_b,Q);
  Q_b = Q - Q_b;

  noalias(w_b) = prod(PI_xtn_ytn,x_xtn_ytn);
  w_b = prod(GT,w_b);
  w_b = prod(Q,w_b);
  w_b *= -1.0;

  /* filter backwards */
  aux::vector x_xtnm1_ytn(N);
  aux::symmetric_matrix P_xtnm1_ytn(N,N);

  noalias(x_xtnm1_ytn) = prod(A_b,x_xtn_ytn) + prod(G_b,w_b);
  noalias(X) = prod(A_b,P_xtn_ytn);
  noalias(P_xtnm1_ytn) = prod(X,trans(A_b));
  noalias(X) = prod(G_b,Q_b);
  noalias(P_xtnm1_ytn) += prod(X,trans(G_b));

  return aux::GaussianPdf(x_xtnm1_ytn, P_xtnm1_ytn);
}

aux::GaussianPdf LinearModel::p_xtnm1_ytnm1(
    const aux::GaussianPdf& p_xtnm1_ytn, const aux::vector& ytnm1,
    const unsigned int delta) {
  return p_xtnp1_ytnp1(p_xtnm1_ytn, ytnm1, delta);
}
