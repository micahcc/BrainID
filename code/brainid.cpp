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
#include "BoldModel.hpp"

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;
    
const int NUM_PARTICLES = 10000;

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



int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

//    aux::symmetric_matrix cov(1);
//    cov(0,0) = 10;
//    aux::GaussianPdf rng1(aux::zero_vector(1), cov);
//    
//    for(int i = 0 ; i<1000 ; i++) {
//        fprintf(stderr, "%8.8f\n", rng1.sample()[0]);
//    }
//
//    cov(0,0) = .001;
//    aux::GaussianPdf rng2(aux::zero_vector(1), cov);
//
//    aux::vector location(1);
//    location[0] = NAN/.001;
//    printf("density at NaN/.001: %e\n", rng1.densityAt(location));
//    location[0] = INFINITY/.001;
//    printf("density at inf/.001: %e\n", rng1.densityAt(location));
//    
//    location(0) = 1;
//    printf("density at 1: %e\n", rng1.densityAt(location));
//    location(0) = 10;
//    printf("density at 10: %e\n", rng1.densityAt(location));
//    location(0) = 100;
//    printf("density at 100: %e\n", rng1.densityAt(location));
//
//    for(int i = 0 ; i<1000 ; i++) {
//        fprintf(stderr, "%8.8f\n", rng2.sample()[0]);
//    }
//
//    fflush(stderr);
//
//    return 0;

    if(argc != 2) {
        fprintf(stderr, "Usage: %s <inputname>\n", argv[0]);
        return -1;
    }
    
    /* Open up the input */
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName( argv[1] );
    reader->Update();

    /* Create the iterator, to move forward in time for a particlular section */
    itk::ImageLinearIteratorWithIndex<ImageType> 
        iter(reader->GetOutput(), reader->GetOutput()->GetRequestedRegion());
    iter.SetDirection(1);
    ImageType::IndexType index;
    index[0] = 1; //skip section label
    index[1] = 5;
    iter.SetIndex(index);

    /* Create a model */
    BoldModel model; 
    aux::GaussianPdf prior = model.suggestPrior();
    prior.sample();
    prior.sample();
    prior.sample();
    aux::DiracMixturePdf x0(prior, NUM_PARTICLES);

    /* Create the filter */
    indii::ml::filter::ParticleFilter<double> filter(&model, x0);
  
    /* create resamplers */
    indii::ml::filter::StratifiedParticleResampler resampler(NUM_PARTICLES);
//    RegularizedParticleResampler resampler_reg(NUM_PARTICLES);
  
    /* estimate and output results */
    aux::vector meas(BoldModel::MEAS_SIZE);
    aux::DiracMixturePdf pred(BoldModel::SYSTEM_SIZE);
    aux::vector mu(BoldModel::SYSTEM_SIZE);

    pred = filter.getFilteredState();
    mu = pred.getDistributedExpectation();
  
    std::ofstream fmeas("ParticleFilterHarness_meas.out");
    std::ofstream fpred("ParticleFilterHarness_filter.out");
    
    double t = 2;
    
    fmeas << "# Created by brainid" << endl;
    fmeas << "# name: measured" << endl;
    fmeas << "# type: matrix" << endl;
    fmeas << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] - 1<< endl;
    fmeas << "# columns: 2" << endl;

    fpred << "# Created by brainid" << endl;
    fpred << "# name: measured" << endl;
    fpred << "# type: matrix" << endl;
    fpred << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] -1 << endl;
    fpred << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;
    
    fpred << t << ' ';
    outputVector(fpred, mu);
    fpred << endl;

    aux::vector sample_state(BoldModel::SYSTEM_SIZE);

//TODO, get resample working
//TODO, get distribution creation working
//TODO, stop using ABS for s(0)
//    aux::vector expectation;
//    pred = filter.getFilteredState();
//    std::vector<aux::DiracPdf> mixdata = pred.getAll();
//    fprintf(stderr, "# number returned: %u\n", mixdata.size());
//    for(int i = 0 ; i<1000 ; i++) {
//        expectation = mixdata[i].getExpectation();
//        outputVector(std::cerr, expectation);
//        fprintf(stderr, "\n");
//    }
//    fflush(stderr);
//    return -1;
//    filter.resample(&resampler);
    while(!iter.IsAtEndOfLine()) {
        fprintf(stderr, "Size0: %u\n", pred.getSize());
        meas(0) = iter.Get();
        ++iter;
    
        std::cerr << "Time " << t << endl;
        filter.filter(t, meas);

        fprintf(stderr, "Size1: %u\n", pred.getSize());
        filter.resample(&resampler);
        pred = filter.getFilteredState();
        fprintf(stderr, "Size2: %u\n", pred.getSize());
        mu = pred.getDistributedExpectation();

        /* output measurement */
        fmeas << t << ' ';
        outputVector(fmeas, meas);
        fmeas << endl;

        /* output filtered state */
        fpred << t << ' ';
        outputVector(fpred, mu);
        fpred << ' ';
        outputVector(fpred, sample_state);
        fpred << endl;
        t += 2;
    }

    fmeas.close();
    fpred.close();

  return 0;

}

