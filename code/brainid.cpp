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

#include "BoldModel.hpp"

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;
    
const int NUM_PARTICLES = 10000;
const int RESAMPNESS = 300;
const double SAMPLERATE = 2;
const int DIVIDER = 8;

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
    index[1] = 1; //skip section label
    index[0] = 5; //just kind of picking a section
    iter.SetIndex(index);

    /* Create a model */
    BoldModel model; 
    aux::DiracMixturePdf x0(NUM_PARTICLES);
    model.generatePrior(x0, NUM_PARTICLES);

    /* Create the filter */
    indii::ml::filter::ParticleFilter<double> filter(&model, x0);
  
    /* create resamplers */
    /* Normal resampler, used to eliminate particles */
    indii::ml::filter::StratifiedParticleResampler resampler(NUM_PARTICLES);

    /* AdditiveNoise */
    indii::ml::filter::AdditiveNoiseParticleResampler< aux::GaussianPdf > resampler_reg();

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
    
    while(!iter.IsAtEndOfLine()) {
        meas(0) = iter.Get();
        ++iter;
   
        std::cerr << "t=" << t << " | iter pos: " << iter.GetIndex()[1] << endl;
        for(int i=0 ; i<DIVIDER ; i++) {
            t += SAMPLERATE/DIVIDER + .0001;//ensure error is +
            filter.filter(t, meas);
        }
        t = (double)((int)t); //round down to remove error
        
        cerr << "ESS: " << filter.getFilteredState().calculateDistributedEss() << endl;
        if(filter.getFilteredState().calculateDistributedEss() < RESAMPNESS) {
            cerr << "Resampling" << endl;
            filter.resample(&resampler);
            filter.resample(&resampler_reg);
        }
       
        //filter.resample(&resampler_reg);
        pred = filter.getFilteredState();
        mu = pred.getDistributedExpectation();

        /* output measurement */
        fmeas << t << ' ';
        outputVector(fmeas, meas);
        fmeas << model.measure(mu)(0) << endl;

        /* output filtered state */
        fpred << t << ' ';
        outputVector(fpred, mu);
//        fpred << ' ';
//        outputVector(fpred, sample_state);
        fpred << endl;
    }
    printf("Index at end: %ld %ld \n", iter.GetIndex()[0], iter.GetIndex()[1]);

    fmeas.close();
    fpred.close();

  return 0;

}

