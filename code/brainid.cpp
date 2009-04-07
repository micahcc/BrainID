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

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "BoldModel.hpp"
#include "RegularizedParticleResamplerMod.hpp"

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

namespace aux = indii::ml::aux;
    
const int NUM_PARTICLES = 10000;
const int RESAMPNESS = 8000;
const double SAMPLERATE = 2;
const int DIVIDER = 32;//divider must be a power of 2 (2, 4, 8, 16, 32....)

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    if(argc < 4) {
        fprintf(stderr, "Usage: %s <inputname> <stimfile> <serialout> [serialin]\n", argv[0]);
        printf("stimfile is very simple, a double time followed by the new value at that time\n");
        return -1;
    }
    
    std::ifstream fin(argv[2]);
    
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
    index[0] = 0; //just kind of picking a section
    iter.SetIndex(index);

    /* Create a model */
    BoldModel model; 
    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    if(argc == 4) {
        model.generatePrior(x0, NUM_PARTICLES);
    } else  {
        std::ifstream serialin(argv[4], std::ios::binary);
        boost::archive::binary_iarchive inArchive(serialin);
        inArchive >> x0;
    }

    /* Create the filter */
    indii::ml::filter::ParticleFilter<double> filter(&model, x0);
  
    /* create resamplers */
    /* Normal resampler, used to eliminate particles */
    indii::ml::filter::StratifiedParticleResampler resampler(NUM_PARTICLES);

    /* Regularized Resample */
    aux::Almost2Norm norm;
    aux::AlmostGaussianKernel kernel(BoldModel::SYSTEM_SIZE, 1);
    RegularizedParticleResamplerMod< aux::Almost2Norm, 
                aux::AlmostGaussianKernel > resampler_reg(norm, kernel);
  
    /* estimate and output results */
    aux::vector meas(BoldModel::MEAS_SIZE);
    aux::DiracMixturePdf pred(BoldModel::SYSTEM_SIZE);
    aux::vector mu(BoldModel::SYSTEM_SIZE);
    aux::symmetric_matrix cov(BoldModel::SYSTEM_SIZE);

    pred = filter.getFilteredState();
    mu = pred.getDistributedExpectation();
  
    std::ofstream fmeas("meas.out");
    std::ofstream fstate("state.out");
    std::ofstream fpart("particles.out");
    
    double t = 0;
    
    fmeas << "# Created by brainid" << endl;
    fmeas << "# name: bold" << endl;
    fmeas << "# type: matrix" << endl;
    fmeas << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] - 1<< endl;
    fmeas << "# columns: 3" << endl;

    fstate << "# Created by brainid" << endl;
    fstate << "# name: states " << endl;
    fstate << "# type: matrix" << endl;
    fstate << "# rows: " << reader->GetOutput()->GetRequestedRegion().GetSize()[1] -1 << endl;
    fstate << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;
    
    fpart << "# Created by brainid" << endl;
    
    aux::vector sample_state(BoldModel::SYSTEM_SIZE);

    bool dirty = false;
    std::vector<aux::DiracPdf> particles;
    aux::vector input(1);
    input[0] = 0;
    double nextinput;
    fin >> nextinput;
    while(!iter.IsAtEndOfLine()) {
        fpart << "# name: particles" << t*1000 << endl;
        fpart << "# type: matrix" << endl;
        fpart << "# rows: " << NUM_PARTICLES << endl;
        fpart << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;
        particles = filter.getFilteredState().getAll();
        for(unsigned int i=0 ; i<particles.size(); i++) {
            fpart << i << " ";
            outputVector(fpart, particles[i].getExpectation());
            fpart << endl;
        }
        fpart << endl;
        
        if(!fin.eof() && t >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
            model.setinput(input);
        }

        if(fmod(t, SAMPLERATE) < 0.01) { 
            ++iter;//intentionally skips first measurement
            meas(0) = iter.Get();
            filter.filter(t,meas);
            double ess = filter.getFilteredState().calculateDistributedEss();
            cerr << "t= " << t << " ESS: " << ess << endl;
            if(ess < RESAMPNESS || isnan(ess) || dirty) {
                cerr << "Resampling" << endl;
                filter.resample(&resampler);
                filter.resample(&resampler_reg);
                dirty = false;
            } else {
                cerr << "No Resampling Necessary!" << endl;
            }
        } else {
            cerr << "t= " << t << endl;
            filter.filter(t);
        }
       
        mu = filter.getFilteredState().getDistributedExpectation();
        cov = filter.getFilteredState().getDistributedCovariance();

        /* output measurement */
        fmeas << t << ' ';
        outputVector(fmeas, meas);
        fmeas << " " << model.measure(mu)(0) << endl;

        /* output filtered state */
        fstate << t << ' ';
        outputVector(fstate, mu);
        
//        outputMatrix(std::cerr, cov);
//        fstate << ' ';
//        outputVector(fstate, sample_state);
        fstate << endl;
        t += SAMPLERATE/DIVIDER; 
    }
    printf("Index at end: %ld %ld \n", iter.GetIndex()[0], iter.GetIndex()[1]);

    fmeas.close();
    fstate.close();

    //serialize

    std::ofstream serialout("distribution.serial", std::ios::binary);
    boost::archive::binary_oarchive outArchive(serialout);
    outArchive << filter.getFilteredState();

  return 0;

}

