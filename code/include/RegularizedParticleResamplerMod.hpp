#ifndef REGULARISEDPARTICLERESAMPLERMOD_HPP
#define REGULARISEDPARTICLERESAMPLERMOD_HPP

#include <indii/ml/filter/ParticleResampler.hpp>
#include <indii/ml/aux/Almost2Norm.hpp>
#include <indii/ml/aux/AlmostGaussianKernel.hpp>

#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_symmetric.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

//this is a reverse dependency, which I know is a bit of a faux pa, but
//it this is already going to be hackish to avoid getting negative values
//for which negative variables are impossible.
#include "BoldModel.hpp"

#include <assert.h>                                                                                      

/**
 * Regularised particle resampler with a few modifications
 * (also with a 10% reduction of the queens english, and I am sure
 * additional ugliness).
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @modder Micah Chambers <micahc@vt.edu>
 * @version $Rev: 576-r1 $
 *
 * Adds standardised kernel noise to each particle. Another resampler, such
 * as DeterministicParticleResampler, should usually be applied first.
 *
 * @param NT Norm type.
 * @param KT Kernel type.
 */
template <class NT = indii::ml::aux::Almost2Norm,
            class KT = indii::ml::aux::AlmostGaussianKernel>
class RegularizedParticleResamplerMod : public indii::ml::filter::ParticleResampler 
{

public:
    /**
     * Constructor.
     *
     * @param N The kernel density norm.
     * @param K The kernel density kernel.
     * @param model the model that this resampling is being done on.
     * TODO change BoldModel to a base class (this would require modifying the base
     * class to include reweight)
     */
    RegularizedParticleResamplerMod(const NT& N, const KT& K, 
                BoldModel* model);

    /**
     * Destructor.
     */
    virtual ~RegularizedParticleResamplerMod();

    /**
     * Set the kernel density norm.
     *
     * @param N The kernel density norm.
     */
    void setNorm(const NT& N);

    /**
     * Set the kernel density kernel.
     *
     * @param K The kernel density kernel.
     */
    void setKernel(const KT& K);

    /**
     * Resample the distribution.
     *
     * @return The resampled distribution.
     */
    virtual indii::ml::aux::DiracMixturePdf resample(
            indii::ml::aux::DiracMixturePdf& p);

    virtual indii::ml::aux::DiracMixturePdf resample(
            indii::ml::aux::DiracMixturePdf& p,
            aux::matrix covariance);

protected:
    /**
     * \f$\|\mathbf{x}\|_p\f$; the norm.
     */
    NT N;

    /**
     * \f$K(\|\mathbf{x}\|_p) \f$; the density kernel.
     */
    KT K;

    /**
     * Used for reality check purposes.
     */
    BoldModel* model;

    /** 
     * Resample Helper - makes it possible to have a default argument for cov.
     */
    virtual void resample_help(indii::ml::aux::DiracMixturePdf& p,
                aux::matrix covariance, indii::ml::aux::DiracMixturePdf& out);
};

//#include "../aux/KernelDensityMixturePdf.hpp"
//#include "../aux/KDTree.hpp"

template <class NT, class KT>
RegularizedParticleResamplerMod<NT,KT>::RegularizedParticleResamplerMod(
            const NT& N, const KT& K, 
            BoldModel* model) : 
            N(N), K(K), model(model)
{
  //
}

template <class NT, class KT>
RegularizedParticleResamplerMod<NT,KT>::~RegularizedParticleResamplerMod() 
{
  //
}

template <class NT, class KT>
void RegularizedParticleResamplerMod<NT,KT>::setNorm(const NT& N) 
{
    this->N = N;
}

template <class NT, class KT>
void RegularizedParticleResamplerMod<NT,KT>::setKernel(const KT& K) 
{
    this->K = K;
}

#include "BoldModel.hpp"
#include <iostream>

template <class NT, class KT>
indii::ml::aux::DiracMixturePdf 
RegularizedParticleResamplerMod<NT, KT>::resample(
        indii::ml::aux::DiracMixturePdf& p)
{
    indii::ml::aux::DiracMixturePdf out(p.getDimensions());
    resample_help(p, p.getDistributedCovariance(), out);
    return out;
}

template <class NT, class KT>
indii::ml::aux::DiracMixturePdf 
RegularizedParticleResamplerMod<NT, KT>::resample(
        indii::ml::aux::DiracMixturePdf& p,
        aux::matrix covariance)
{
    indii::ml::aux::DiracMixturePdf out(p.getDimensions());
    resample_help(p, covariance, out);
    return out;
}

template <class NT, class KT>
void
RegularizedParticleResamplerMod<NT, KT>::resample_help(
            indii::ml::aux::DiracMixturePdf& p,
            aux::matrix sd, indii::ml::aux::DiracMixturePdf& out)
{
    boost::mpi::communicator world;
//    const unsigned int rank = world.rank();
    namespace aux = indii::ml::aux;
    namespace ublas = boost::numeric::ublas;
    namespace lapack = boost::numeric::bindings::lapack;

    aux::vector x(p.getDimensions());
    double weight = 0;

    size_t badcount = 0;
    /* rebuild distribution with kernel noise */
    for (unsigned int i = 0; i < p.getSize(); i++) {
        noalias(x) = p.get(i) + prod(sd, K.sample() * N.sample(
                    p.getDimensions()));
        weight = p.getWeight(i);
        if(model && model->reweight(x, weight)) 
            badcount++;
        out.add(x, weight);
    }
    if(badcount != 0) {
        std::cerr << "Reset " << badcount << "/" << p.getSize() << "particles"
                    << std::endl;
    }
};

#endif

