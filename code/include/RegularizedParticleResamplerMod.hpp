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

private:
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
RegularizedParticleResamplerMod<NT, KT>::resample(indii::ml::aux::DiracMixturePdf& p)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    namespace aux = indii::ml::aux;
    namespace ublas = boost::numeric::ublas;
    namespace lapack = boost::numeric::bindings::lapack;

    aux::DiracMixturePdf r(p.getDimensions());
    aux::vector x(p.getDimensions());
    double weight = 0;

    /* standardise particles */
    aux::matrix sd(p.getDistributedCovariance());
    aux::vector diag_v(sd.size1());
//    std::cout << "STARTING!" << std::endl;
//    outputMatrix(std::cout, sd);
//    std::cout << std::endl << std::endl;;
    int err = lapack::syev('V', 'U', sd, diag_v);
    assert(err == 0);

    for(unsigned int i = 0 ; i<diag_v.size() ; i++) {
        if(diag_v(i) < 0) {
            std::cerr << "ERROR diagnal matrix gives non-real solution" << std::endl;
            //outputMatrix(std::cerr, p.getDistributedCovariance());
            //exit(-1);
            diag_v(i) = 0;
        }
    }

    diag_v = element_sqrt(diag_v);
    ublas::diagonal_matrix<double, ublas::column_major, 
                ublas::unbounded_array<double> > diag_dm(diag_v.size(), diag_v.data());
    aux::matrix diag_m(diag_dm);
    
//    outputMatrix(std::cout, sd);
//    std::cout << std::endl << std::endl;;
//    outputVector(std::cout, diag_v);
//    std::cout << std::endl << std::endl;;
//    outputMatrix(std::cout, diag_m);
//    std::cout << std::endl << std::endl;;
    aux::matrix tmp = prod(sd,diag_m);
    sd = prod(tmp, trans(sd));
//    tmp = prod(sd, sd);
//    std::cout << "This matrix should be equal to the first one printed" << std::endl;
//    outputMatrix(std::cout, tmp);
    size_t badcount = 0;
    /* rebuild distribution with kernel noise */
    for (unsigned int i = 0; i < p.getSize(); i++) {
        noalias(x) = p.get(i) + prod(sd, K.sample() * N.sample(
                    p.getDimensions()));
        weight = p.getWeight(i);
        if(model != NULL) {
            if(model->reweight(x, weight))
                badcount++;
            
        }
        r.add(x, weight);
    }
    if(badcount != 0) {
        std::cerr << "Reweighted " << badcount << "/" << p.getSize() << "particles"
                    << std::endl;
    }

    return r;
};

#endif

