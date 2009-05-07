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
#include <boost/program_options.hpp>

#include "BoldModel.hpp"
#include "RegularizedParticleResamplerMod.hpp"

#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

namespace opts = boost::program_options;
namespace aux = indii::ml::aux;
    
const double RESAMPNESS = .8; //should be some percentage of NUM_PARTICLES
const double SAMPLETIME = 2; //in seconds, should get from fmri image
//const int DIVIDER = 8;//divider must be a power of 2 (2, 4, 8, 16, 32....)

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  4 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

// Declare the supported options.
// po::options_description desc("Allowed options");
// desc.add_options()
//     ("help", "produce help message")
//         ("compression", po::value<int>(), "set compression level")
//         ;
//
//         po::variables_map vm;
//         po::store(po::parse_command_line(ac, av, desc), vm);
//         po::notify(vm);    
//


int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    int NUM_PARTICLES;
    int DIVIDER;

    //CLI
    opts::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("particles,p", opts::value<int>(), "Number of particles to use.")
            ("timeseries,t", opts::value<string>(), "2D timeseries file")
            ("divider,d", opts::value<int>(), "Ratio of linearization points to number of samples")
            ("stimfile,s", opts::value<string>(), "file containing \"time value\""
                        "pairs which give the time at which input changed")
            ("serialout", opts::value<string>(), "Where to put a serial output file")
            ("serialin", opts::value<string>(), "Where to find a serial input file");

    opts::variables_map cli_vars;
    opts::store(opts::parse_command_line(argc, argv, desc), cli_vars);
    opts::notify(cli_vars);
    
    if(cli_vars.count("help")) {
        cout << desc << endl;
        return 1;
    }
    
    if(cli_vars.count("divider")) {
        DIVIDER = cli_vars["divider"].as < int >();
        cout << "Setting divider to" << DIVIDER << endl
                    << "Setting timestep to " << 2./DIVIDER << endl;
    } else {
        cout << "Setting divider to 8" << endl;
        DIVIDER = 8;
    }
    
    if(cli_vars.count("particles")) {
        cout << "Number of Particles: " << cli_vars["particles"].as< int >() << endl;
        NUM_PARTICLES = cli_vars["particles"].as < int >();
    } else {
        cout << "Need to enter a number of particles" << endl;
        return -1;
    }

    if(cli_vars.count("timeseries")) {
        cout << "Timeseries: " << cli_vars["timeseries"].as< string >() << endl;
    } else {
        cout << "Need to enter a timeseries file" << endl;
        return -1;
    }

    if(cli_vars.count("stimfile")) {
        cout << "Stimfile: " << cli_vars["stimfile"].as < string >() << endl;
    } else {
        cout << "Need to enter a stimulus input file" << endl;
        return -2;
    }

    if(cli_vars.count("serialout")) {
        cout << "Serial Output: " << cli_vars["serialout"].as < string >() << endl;
    } else {
        cout << "No serial out selected" << endl;
    }

    if(cli_vars.count("serialin")) {
        cout << "Serial Input: " << cli_vars["serialin"].as < string >() << endl;
    } else {
        cout << "No serial input selected" << endl;
    }

    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    std::ifstream fin;
    
    /* Open up the input */
    ImageReaderType::Pointer reader;
    itk::ImageLinearIteratorWithIndex<ImageType> iter;
    if(rank == 0) {
        reader = ImageReaderType::New();
        reader->SetFileName( cli_vars["timeseries"].as< string >() );
        reader->Update();

        /* Create the iterator, to move forward in time for a particlular section */
        iter = itk::ImageLinearIteratorWithIndex<ImageType>(reader->GetOutput(), 
                    reader->GetOutput()->GetRequestedRegion());
        iter.SetDirection(3);
        ImageType::IndexType index;
        index[3] = 1; //skip section label
        index[2] = 0; //skip section label
        index[1] = 0; //skip section label
        index[0] = 0; //just kind of picking a section
        iter.SetIndex(index);
    }

    /* Create a model */
    BoldModel model; 
    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    if(cli_vars.count("serialin")) {
        std::ifstream serialin(cli_vars["serialin"].as< string >().c_str(), std::ios::binary);
        boost::archive::binary_iarchive inArchive(serialin);
        inArchive >> x0;
    } else  {
        model.generatePrior(x0, NUM_PARTICLES / size);
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
  
    /* output setup */
    std::ofstream fmeas;
    std::ofstream fstate;
    std::ofstream fcov;
#ifdef OUTPART
    std::ofstream fpart;
#endif //OUTPART

    if(rank == 0) {
        std::ostringstream iss("");
        iss << "meas" << ".out";
        fmeas.open(iss.str().c_str());
        
        iss.str("");
        iss << "state" << ".out";
        fstate.open(iss.str().c_str());
        
        iss.str("");
        iss << "cov" << ".out";
        fcov.open(iss.str().c_str());

#ifdef OUTPART
       iss.str("");
       iss << "particles" << rank << ".out";
       fpart.open(iss.str().c_str());
#endif //OUTPART
        
        fmeas << "# Created by brainid" << endl;
        fmeas << "# name: bold" << endl;
        fmeas << "# type: matrix" << endl;
        fmeas << "# rows: " << 
                    (reader->GetOutput()->GetRequestedRegion().GetSize()[3] - 1)
                    << endl;
        fmeas << "# columns: 3" << endl;

        fstate << "# Created by brainid" << endl;
        fstate << "# name: states " << endl;
        fstate << "# type: matrix" << endl;
        fstate << "# rows: " << 
                    (reader->GetOutput()->GetRequestedRegion().GetSize()[3] -1) 
                    << endl;
        fstate << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;

        fcov << "# Created by brainid" << endl;
        fcov << "# name: covariances" << endl;
        fcov << "# type: matrix" << endl;
        fcov << "# ndims: 3" << endl;
        fcov <<  BoldModel::SYSTEM_SIZE << " " << BoldModel::SYSTEM_SIZE << " "
                    << reader->GetOutput()->GetRequestedRegion().GetSize()[3] -1 << endl;

#ifdef OUTPART
        fpart << "# Created by brainid" << endl;
#endif //OUTPART
    } 

#ifdef OUTPART
    std::vector<aux::DiracPdf> particles;
#endif //OUTPART

    /* Simulation Section */
    aux::vector input(1);
    aux::vector meas(1);
    aux::vector mu(BoldModel::SYSTEM_SIZE);
    aux::symmetric_matrix cov(BoldModel::SYSTEM_SIZE);
    input[0] = 0;
    double nextinput;
    int disctime = 0;
    bool done = false;
    int tmp = 0;
    if(rank == 0) {
        fin.open(cli_vars["stimfile"].as< string >().c_str());
        fin >> nextinput;
    }

    while(!done) {
#ifdef OUTPART
        if( rank == 0 ) {
            fpart << "# name: particles" << setw(5) << t*10000 << endl;
            fpart << "# type: matrix" << endl;
            fpart << "# rows: " << NUM_PARTICLES << endl;
            fpart << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;
            fpart << "# time: " << time(NULL) << endl;
            particles = filter.getFilteredState().getAll();
            for(unsigned int i=0 ; i<particles.size(); i++) {
                fpart << i << " ";
                outputVector(fpart, particles[i].getExpectation());
                fpart << endl;
            }
            fpart << endl;
        }
#endif // OUTPART
        
        //the +.1 is just to remove the possibility of missing something
        //due to roundoff error, since disctime is the smallest possible 
        //timestep adding .1 will never go into the next timestep
        if(rank == 0 && !fin.eof() && disctime*SAMPLETIME/DIVIDER >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
        }

        boost::mpi::broadcast(world, input, 0);
        model.setinput(input);

        /* time for update */
        if( rank == 0 ) 
            cerr << "t= " << disctime*SAMPLETIME/DIVIDER << ", ";
        
        if(disctime%DIVIDER == 0) {
            if(rank == 0) {
                meas(0) = iter.Get();
                ++iter;
                done = iter.IsAtEndOfLine();
            }
            boost::mpi::broadcast(world, meas, 0);
            boost::mpi::broadcast(world, done, 0);
            filter.filter(disctime*SAMPLETIME/DIVIDER,meas);

            double ess = filter.getFilteredState().calculateDistributedEss();

            if(ess < NUM_PARTICLES*RESAMPNESS || isnan(ess) || isinf(ess)) {
                if( rank == 0 )
                    cerr << endl << " ESS: " << ess << ", Deterministic Resampling" << endl;
                filter.resample(&resampler);
                if( rank == 0)
                    cerr << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter.resample(&resampler_reg);
            } else if (rank == 0 )
                cerr << endl << " ESS: " << ess << ", No Resampling Necessary!" << endl;
        
            mu = filter.getFilteredState().getDistributedExpectation();
            cov = filter.getFilteredState().getDistributedCovariance();
            if( rank == 0 ) {
                /* Get state */
                
                /* output measurement */
                fmeas << setw(10) << disctime*SAMPLETIME/DIVIDER;
                fmeas << setw(10) << input[0];
                fmeas << setw(14) << model.measure(mu)(0) << endl;

                /* output filtered state */
                fstate << setw(10) << disctime*SAMPLETIME/DIVIDER; 
                outputVector(fstate, mu);
                fstate << endl;
                
                /* output filtered covariance */
                for(int ii=0 ; ii<BoldModel::SYSTEM_SIZE; ii++) {
                    for(int jj=0 ; jj<BoldModel::SYSTEM_SIZE; jj++) {
                        fcov << cov(ii,jj) << endl;
                    }
                }
            }

        } else {
            filter.filter(disctime*SAMPLETIME/DIVIDER);
        }
   
       
        /* Update Time, disctime */
        tmp = disctime;
        boost::mpi::broadcast(world, disctime, 0);
        if(tmp != disctime) {
            cerr << "ERROR ranks have gotten out of sync at " << disctime << endl;
            exit(-1);
        }
        disctime++;
    }
    printf("Index at end: %ld %ld \n", iter.GetIndex()[0], iter.GetIndex()[1]);

    //serialize

    x0 = filter.getFilteredState();
    if( rank == 0 ) {
        fmeas.close();
        fstate.close();
        fcov.close();
        
        if(cli_vars.count("serialout")) {
            std::ofstream serialout(cli_vars["serialout"].as< string >().c_str(), std::ios::binary);
            boost::archive::binary_oarchive outArchive(serialout);
            outArchive << x0;
        }
    }

  return 0;

}

