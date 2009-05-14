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
#include <indii/ml/aux/matrix.hpp>

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
#include <fstream>
#include <sstream>

using namespace std;

namespace opts = boost::program_options;
namespace aux = indii::ml::aux;
    
const double SAMPLETIME = 2; //in seconds, should get from fmri image

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  4 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    ostream* out;
    ofstream nullout("/dev/null");

    if(rank == 0) {
        out = &cout;
    } else {
        out = &nullout;
    }

    double RESAMPNESS = .5; //should be some percentage of NUM_PARTICLES
    int NUM_PARTICLES = 60000;
    int DIVIDER = 8;
    int NUM_MOSTPROB= 4;
    aux::vector cheat(BoldModel::SYSTEM_SIZE) ;
    double curweight = 0; //how much to weight the current time vs. old times
    int weightfunc = BoldModel::NORM; //type of weighting function to use

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
            ("serialin", opts::value<string>(), "Where to find a serial input file")
            ("weightf", opts::value<string>(), "weighting function to use, options:"
                        "norm || exp || hyp")
            ("reweight", opts::value<string>(), "how to reweight particles, options:"
                        "mult || <averaging percent of now>")
            ("resampness,r", opts::value<string>(), "Ratio of total particles that the ESS must "
                        "reach for the filter to resample. Ex .8 Ex2. .34") 
            ("cheat", opts::value<double>(), "This cheats and gives the true starting parameters"
                        "to the particle filter. This is just a validation technique for the "
                        "filter. Syntax: \"Tau_s Tau_f Epsilon Tau_0 "
                        "alpha E_0 V_0 v_t0 q_t0 s_t0 f_t0\"");

    opts::variables_map cli_vars;
    opts::store(opts::parse_command_line(argc, argv, desc), cli_vars);
    opts::notify(cli_vars);
    
    ///////////////////////////////////////////////////////////////////////////////
    //Parse command line options
    ///////////////////////////////////////////////////////////////////////////////
    if(cli_vars.count("help")) {
        *out << desc << endl;
        return 1;
    }
    
    if(cli_vars.count("resampness")) {
        RESAMPNESS = cli_vars["resampness"].as < double >();
    } 
    *out << left << setw(20) << "resampness" << ":" << RESAMPNESS << endl;
    
    if(cli_vars.count("divider")) {
        DIVIDER = cli_vars["divider"].as < int >();
    } 
    *out << left << setw(20) << "divider" << ":" << DIVIDER << endl;
    
    
    if(cli_vars.count("particles")) {
        NUM_PARTICLES = cli_vars["particles"].as < int >();
    } 
    *out << left << setw(20) << "Particles" << ":" << NUM_PARTICLES << endl;

    if(cli_vars.count("timeseries")) {
        *out << left << setw(20) <<  "Timeseries" << ":" << cli_vars["timeseries"].as< string >() << endl;
    } else {
        *out << "Error! Timeseries: Need to enter a timeseries file!" << endl;
        return -1;
    }

    if(cli_vars.count("stimfile")) {
        *out << left << setw(20) <<  "Stimfile" << ":" << cli_vars["stimfile"].as < string >() << endl;
    } else {
        *out << "Error! Stimfile: Need to enter a stimulus input file" << endl;
        return -2;
    }

    if(cli_vars.count("serialout")) {
        *out << left << setw(20) << "SerialOut" << ":" << cli_vars["serialout"].as < string >() << endl;
    } 

    if(cli_vars.count("serialin")) {
        *out << left << setw(20) << "SerialIn" << ":" << cli_vars["serialin"].as < string >() << endl;
    } 

    if(cli_vars.count("weightf")) {
        if(cli_vars["weightf"].as<string>().compare("exp") == 0) {
            *out << left << setw(20) << "weightf" << ":Weighting based on the"
                        << " exponential distribution" << endl;
            weightfunc = BoldModel::EXP;
        } else if(cli_vars["weightf"].as<string>().compare("hyp") == 0) {
            *out << left << setw(20) << "weightf" << ":Weighting based on 1/dist" << endl;
            weightfunc = BoldModel::HYP;
        } else {
            *out << left << setw(20) << "weightf" << ":Weighting based on the normal"
                        << " distribution" << endl;
            weightfunc = BoldModel::NORM;
        }
    } else {
        *out << left << setw(20) << "weightf" << ":Weighting based on the normal"
                    << " distribution" << endl;
        weightfunc = BoldModel::NORM;
    }
    
    if(cli_vars.count("reweight")) {
        istringstream iss(cli_vars["rewight"].as<string>());
        if(cli_vars["reweight"].as<string>().compare("mult") == 0) {
            *out << left << setw(20) << "re-weight" << ": will multiply old weight"
                        << " by new weight for updates" << endl;
            curweight = 0;
        } else {
            iss >> curweight;
            *out << left << setw(20) << "reweight" << ": weight update: <new> = <old>*"
                        << (1-curweight) << "+<now>*" << curweight << endl;
        }
    } else {
        curweight = 0;
    }
    
    if(cli_vars.count("cheat")) {
        *out << left << setw(20) << "cheat" << ":Cheating by distributing starting"
                    << " particles around:" 
                    << cli_vars["cheat"].as < string >() << endl;

        istringstream iss(cli_vars["cheat"].as<string>());
        
        for(int i = 0 ; i < BoldModel::SYSTEM_SIZE ; i++) {
            if(iss.eof()) {
                cerr << "Error not enough arguments given on command line" << endl;
                exit(-3);
            }
            iss >> cheat[i];
        }
        
    } 

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
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
    BoldModel model(zero_vector(BoldModel::SYSTEM_SIZE), weightfunc, curweight);
    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    if(cli_vars.count("serialin")) {
        std::ifstream serialin(cli_vars["serialin"].as< string >().c_str(), std::ios::binary);
        boost::archive::binary_iarchive inArchive(serialin);
        inArchive >> x0;
    } else if(cli_vars.count("cheat")) {
        model.generatePrior(x0, NUM_PARTICLES / size, cheat);
    } else {
        *out << "Generating prior" << endl;
        model.generatePrior(x0, NUM_PARTICLES / size);
        aux::matrix tmp = x0.getDistributedCovariance();
        *out << "Covariance: " << endl;
        outputMatrix(*out, tmp);
        *out << endl;
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
    aux::DiracMixturePdf distr(BoldModel::SYSTEM_SIZE);
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
        *out << "t= " << disctime*SAMPLETIME/DIVIDER << ", ";
        
        if(disctime%DIVIDER == 0) { //time for update!
            //acquire the latest measurement
            if(rank == 0) {
                meas(0) = iter.Get();
                ++iter;
                done = iter.IsAtEndOfLine();
            }

            //send meas and done to other nodes
            boost::mpi::broadcast(world, meas, 0);
            boost::mpi::broadcast(world, done, 0);
            
            //step forward in time, with measurement
            filter.filter(disctime*SAMPLETIME/DIVIDER, meas);

            //check to see if resampling is necessary
            double ess = filter.getFilteredState().calculateDistributedEss();
            
            //check for errors, could be caused by total collapse of particles,
            //for instance if all the particles go to an unreasonable value like 
            //inf/nan/neg
            if(isnan(ess) || isinf(ess)) {
                cerr << "Total Weight: " << filter.getFilteredState().getTotalWeight() << endl;
                aux::vector weights = filter.getFilteredState().getWeights();
                outputVector(cerr, weights);
                exit(-5);
            }

            //time to resample
            if(ess < NUM_PARTICLES*RESAMPNESS) {
                *out << endl << " ESS: " << ess << ", Deterministic Resampling" << endl;
                filter.resample(&resampler);
                
                *out << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter.resample(&resampler_reg);
            } else
                *out << endl << " ESS: " << ess << ", No Resampling Necessary!" << endl;
        
            distr = filter.getFilteredState();
            mu = distr.getDistributedExpectation();
            cov = distr.getDistributedCovariance();
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

        } else { //no update available, just step update states
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

