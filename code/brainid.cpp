//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan

#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>

#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/vector.hpp>

#include <indii/ml/aux/Almost2Norm.hpp>
#include <indii/ml/aux/AlmostGaussianKernel.hpp>
#include <indii/ml/filter/RegularisedParticleResampler.hpp>

#include "boost/numeric/bindings/lapack/lapack.hpp"

#include "BoldModel.hpp"

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;
    
const int NUM_PARTICLES = 10000;
const int RESAMPNESS = 300;
const double SAMPLERATE = 2;
const int DIVIDER = 8;//divider must be a power of 2 (2, 4, 8, 16, 32....)

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

//class ParticleFilterMod : public ParticleFilter<double>
//{
//public:
//    void print_particles() {
//        for (unsigned int i = 0; i < this->p_xtn_ytn.getSize(); i++) {
//            outputVector(std::cerr, this->p_xtn_ytn.get(i));
//        }
//    }
//
//    ParticleFilterMod(ParticleFilterModel<double>* model,
//            indii::ml::aux::DiracMixturePdf& p_x0);
//
//    virtual void filter(const double tnp1, const indii::ml::aux::vector& ytnp1);
//};
//
//void ParticleFilterMod::filter(const double tnp1, const aux::vector& ytnp1) {
//    
//};
//
//ParticleFilterMod::ParticleFilterMod(ParticleFilterModel<double>* model,
//            indii::ml::aux::DiracMixturePdf& p_x0) : 
//            ParticleFilter<double>(model, p_x0) {
//
//};

void fix(aux::symmetric_matrix& mat) 
{
    cerr << "Fixing matrix " << endl;
    outputMatrix(cerr, mat);
    unsigned int i, j;
    for (j = 0; j < mat.size2(); j++) {
        for (i = 0; i < mat.size1(); i++) {
            mat(i, j) = mat(i,j) < 0 ? -mat(i,j) : mat(i,j);
        }
    }
    outputMatrix(cerr, mat);
}

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    if(argc != 2) {
        fprintf(stderr, "Usage: %s <inputname>\n", argv[0]);
        return -1;
    }
    
    /* Open up the input */
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    /* Create the iterator, to move forward in time for a particlular section */
    itk::ImageLinearIteratorWithIndex<ImageType> iter(reader->GetOutput(), 
                reader->GetOutput()->GetRequestedRegion());
    iter.SetDirection(1);
    ImageType::IndexType index;
    index[1] = 0; //skip section label later by allowing iter++ on first pass
    index[0] = 5; //just kind of picking a section
    iter.SetIndex(index);

    /* Create a model */
    BoldModel model; 
    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    model.generatePrior(x0, NUM_PARTICLES);

    /* Create the filter */
    indii::ml::filter::ParticleFilter<double> filter(&model, x0);
  
    /* create resamplers */
    /* Normal resampler, used to eliminate particles */
    indii::ml::filter::StratifiedParticleResampler resampler(NUM_PARTICLES);

    /* Regularized Resample */
    aux::Almost2Norm norm;
    aux::AlmostGaussianKernel kernel(BoldModel::SYSTEM_SIZE, 1);
    indii::ml::filter::RegularisedParticleResampler< aux::Almost2Norm, 
                aux::AlmostGaussianKernel > resampler_reg(norm, kernel);
  
    /* estimate and output results */
    aux::vector meas(BoldModel::MEAS_SIZE);
    aux::DiracMixturePdf pred(BoldModel::SYSTEM_SIZE);
    aux::vector mu(BoldModel::SYSTEM_SIZE);

    pred = filter.getFilteredState();
    mu = pred.getDistributedExpectation();
  
    std::ofstream fmeas("meas.out");
    std::ofstream fpred("pred.out");
    std::ofstream fpart("particles.out");
    
    double t = 0;
    
    fmeas << "# Created by brainid" << endl;
    fmeas << "# name: bold" << endl;
    fmeas << "# type: matrix" << endl;
    fmeas << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] - 1<< endl;
    fmeas << "# columns: 3" << endl;

    fpred << "# Created by brainid" << endl;
    fpred << "# name: calc " << endl;
    fpred << "# type: matrix" << endl;
    fpred << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] -1 << endl;
    fpred << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;
    
    fpart << "# Created by brainid" << endl;
    
    aux::vector sample_state(BoldModel::SYSTEM_SIZE);

//TODO, get resample working
//TODO, get distribution creation working
//TODO, stop using ABS for s(0)
    bool dirty = false;
//    std::vector<aux::DiracPdf> particles;
    while(!iter.IsAtEndOfLine()) {
//        fpart << "# name: partilces" << t << endl;
//        fpart << "# type: matrix" << endl;
//        fpart << "# rows: " << NUM_PARTICLES << endl;
//        fpart << "# columns: " << BoldModel::SYSTEM_SIZE << endl;
//        particles = filter.getFilteredState().getAll();
//        for(int i=0 ; i<particles.size(); i++) {
//            outputVector(fpart, particles[i].getExpectation());
//            fpart << endl;
//        }
//        fpart << endl;
        int err;
        if(fmod(t, SAMPLERATE) < 0.01) { 
            ++iter;//intentionally skips first measurement
            meas(0) = iter.Get();
            filter.filter(t,meas);
            double ess = pred.calculateDistributedEss();
            cerr << "t= " << t << " ESS: " << ess << endl;
            if(ess < RESAMPNESS || isnan(ess) || dirty) {
                cerr << "Resampling" << endl;
                symmetric_matrix sigma(filter.getFilteredState().getCovariance());
                err = boost::numeric::bindings::lapack::pptrf(sigma);
                assert(err == 0);
//                filter.resample(&resampler);
                filter.resample(&resampler_reg);
                dirty = false;
            }
        } else {
            filter.filter(t);
        }
       
        mu = filter.getFilteredState().getDistributedExpectation();

        /* output measurement */
        fmeas << t << ' ';
        outputVector(fmeas, meas);
        fmeas << " " << model.measure(mu)(0) << endl;

        /* output filtered state */
        fpred << t << ' ';
        outputVector(fpred, mu);
//        fpred << ' ';
//        outputVector(fpred, sample_state);
        fpred << endl;
        t += SAMPLERATE/DIVIDER; 
    }
    printf("Index at end: %ld %ld \n", iter.GetIndex()[0], iter.GetIndex()[1]);

    fmeas.close();
    fpred.close();

  return 0;

}

