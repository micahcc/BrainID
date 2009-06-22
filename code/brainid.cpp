//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan

#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkMetaDataObject.h>

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

#include "modNiftiImageIO.h"
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
    
double SAMPLETIME = 2; //in seconds, should get from fmri image

/* Typedefs */
typedef double ImagePixelType;
typedef itk::Image< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

/* Globals from Command Line */
double RESAMPNESS    = .5; //should be some percentage of NUM_PARTICLES
size_t NUM_PARTICLES = 60000;
int    DIVIDER       = 8;
double curweight     = 0; //how much to weight the current time vs. old times
int weightfunc       = BoldModel::NORM; //type of weighting function to use

string serialofile   = "";
string serialifile   = "";
string stimfile      = "";
string seriesfile    = "";
string boldfile      = "";
string statefile     = "";
string covfile       = "";
string partfile      = "";

bool cheating;
aux::vector cheat(BoldModel::SYSTEM_SIZE) ;

void writeVector(Image4DType::Pointer out, int dir, const aux::vector& input, 
            Image4DType::IndexType start)
{
    itk::ImageLinearIteratorWithIndex<Image4DType> 
                it(out, out->GetRequestedRegion());
    it.SetDirection(dir);
    it.SetIndex(start);

    size_t i;
    for(i = 0 ; i < input.size() && !it.IsAtEndOfLine() ; i++) {
        it.Set(input[i]);
        ++it;
    }

    assert(i==input.size() && it.IsAtEndOfLine());
}

void writeParticles(Image4DType::Pointer out, 
            const std::vector<aux::DiracPdf>& elems, int t)
{
    Image4DType::IndexType pos = {{0, 0, 0, t}};
    
    for(size_t i = 0 ; i<elems.size() ; i++) {
        pos[2] = i;
        writeVector(out, 1, elems[i].getExpectation(), pos);
    }
}

//dir1 should be the first matrix dimension, dir2 the second
void writeMatrix(Image4DType::Pointer out, int dir1, int dir2, 
            const aux::matrix& input, Image4DType::IndexType start)
{
    itk::ImageSliceIteratorWithIndex< Image4DType > 
                it(out, out->GetRequestedRegion());
    it.SetFirstDirection(dir1);
    it.SetSecondDirection(dir2);
    
    it.SetIndex(start);

    for(size_t j = 0 ; j < input.size2() && !it.IsAtEndOfSlice() ; j++) {
        for(size_t i = 0 ; i < input.size1() && !it.IsAtEndOfLine() ; i++) {
            it.Set(input(i,j));
        }
        it.NextLine();
    }
}

void readVector(const Image4DType::Pointer in, int dir, aux::vector& input, 
            Image4DType::IndexType start)
{
    itk::ImageLinearConstIteratorWithIndex<Image4DType> 
                it(in, in->GetRequestedRegion());
    it.SetDirection(dir);
    it.SetIndex(start);

    size_t i;
    for(i = 0 ; i < input.size() && !it.IsAtEndOfLine() ; i++) {
        input[i] = it.Get();
    }

    assert(i==input.size());
}

/* init4DImage 
 * sets the ROI for a new image and then calls allocate
 */
void init4DImage(Image4DType::Pointer& out, size_t xlen, size_t ylen, 
            size_t zlen, size_t tlen)
{
    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index;
    Image4DType::SizeType out_size;

    out_size[0] = xlen;
    out_size[1] = ylen;
    out_size[2] = zlen;
    out_size[3] = tlen; 
    
    out_index[0] = 0;
    out_index[1] = 0;
    out_index[2] = 0;
    out_index[3] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    out->SetRegions( out_region );
    out->Allocate();
}

/* Parses the Command line and fills in the globals */
void parse_cli(int argc, char* argv[])
{
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
            ("resampness,r", opts::value<double>(), "Ratio of total particles that the ESS must "
                        "reach for the filter to resample. Ex .8 Ex2. .34") 
            ("cheat", opts::value<double>(), "This cheats and gives the true starting parameters"
                        "to the particle filter. This is just a validation technique for the "
                        "filter. Syntax: \"Tau_s Tau_f Epsilon Tau_0 "
                        "alpha E_0 V_0 v_t0 q_t0 s_t0 f_t0\"")
            ("boldout,b", opts::value<string>(), "Name of image file to write bold data to")
            ("stateout,o", opts::value<string>(), "Name of image file to write" 
                        " state variable data to")
            ("covout,c", opts::value<string>(), "Name of image file to write covariance data to");

    opts::variables_map cli_vars;
    try {
        opts::store(opts::parse_command_line(argc, argv, desc), cli_vars);
        opts::notify(cli_vars);
    } catch(...) {
        cout << "Improper Command Line Option Given!" << endl << endl;
        cout << desc << endl;
        exit(-6);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    //Parse command line options
    ///////////////////////////////////////////////////////////////////////////////
    if(cli_vars.count("help")) {
        cout << desc << endl;
        exit(1);
    }
    
    if(cli_vars.count("resampness")) {
        RESAMPNESS = cli_vars["resampness"].as < double >();
    } 
    cout << left << setw(20) << "resampness" << ":" << RESAMPNESS << endl;
    
    if(cli_vars.count("divider")) {
        DIVIDER = cli_vars["divider"].as < int >();
    } 
    cout << left << setw(20) << "divider" << ":" << DIVIDER << endl;
    
    if(cli_vars.count("particles")) {
        NUM_PARTICLES = cli_vars["particles"].as < int >();
    } 
    cout << left << setw(20) << "Particles" << ":" << NUM_PARTICLES << endl;

    if(cli_vars.count("timeseries")) {
        seriesfile = cli_vars["timeseries"].as< string >();
        cout << left << setw(20) <<  "Timeseries" << ":" << seriesfile << endl;
    } else {
        cout << "Error! Timeseries: Need to enter a timeseries file!" << endl;
        exit(-1);
    }
    
    if(cli_vars.count("stimfile")) {
        stimfile = cli_vars["stimfile"].as< string >();
        cout << left << setw(20) <<  "Stimfile" << ":" << stimfile << endl;
    } else {
        cout << left << setw(20) <<  "Stimefile" << ": None! Will decay freely" << endl;
    }

    if(cli_vars.count("serialout")) {
        serialofile = cli_vars["serialout"].as< string >();
        cout << left << setw(20) << "SerialOut" << ":" << serialofile << endl;
    } 
    
    if(cli_vars.count("serialin")) {
        serialifile = cli_vars["serialin"].as< string >();
        cout << left << setw(20) << "SerialIn" << ":" << serialifile << endl;
    }
    
    if(cli_vars.count("weightf")) {
        if(cli_vars["weightf"].as<string>().compare("exp") == 0) {
            cout << left << setw(20) << "weightf" << ":Weighting based on the"
                        << " exponential distribution" << endl;
            weightfunc = BoldModel::EXP;
        } else if(cli_vars["weightf"].as<string>().compare("hyp") == 0) {
            cout << left << setw(20) << "weightf" << ":Weighting based on 1/dist" << endl;
            weightfunc = BoldModel::HYP;
        } else {
            cout << left << setw(20) << "weightf" << ":Weighting based on the normal"
                        << " distribution" << endl;
            weightfunc = BoldModel::NORM;
        }
    } else {
        cout << left << setw(20) << "weightf" << ":Weighting based on the normal"
                    << " distribution" << endl;
        weightfunc = BoldModel::NORM;
    }
    
    if(cli_vars.count("reweight")) {
        istringstream iss(cli_vars["rewight"].as<string>());
        if(cli_vars["reweight"].as<string>().compare("mult") == 0) {
            cout << left << setw(20) << "re-weight" << ": will multiply old weight"
                        << " by new weight for updates" << endl;
            curweight = 0;
        } else {
            iss >> curweight;
            cout << left << setw(20) << "reweight" << ": weight update: <new> = <old>*"
                        << (1-curweight) << "+<now>*" << curweight << endl;
        }
    } else {
        curweight = 0;
    }
    
    cheating = false;
    if(cli_vars.count("cheat")) {
        cheating = true;
        cout << left << setw(20) << "cheat" << ":Cheating by distributing starting"
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
    
    if(cli_vars.count("boldout")) {
        boldfile = cli_vars["boldout"].as<string>();
        cout << left << setw(20) << "boldout" << ":" << boldfile << endl;
    }

    if(cli_vars.count("stateout")) {
        statefile = cli_vars["stateout"].as<string>();
        cout << left << setw(20) << "stateout" << ":" << statefile << endl;
    } 
    
    if(cli_vars.count("covout")) {
        covfile = cli_vars["covout"].as<string>();
        cout << left << setw(20) << "covout" << ":" << covfile << endl;
    } 
    
    if(cli_vars.count("partout")) {
        partfile = cli_vars["partout"].as<string>();
        cout << left << setw(20) << "" << ":" << covfile << endl;
        cout << "WARNING saving all the particles can make a HUGE file" << endl;
    } 
}


/* Main Function */
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
        parse_cli(argc, argv);
    } else {
        out = &nullout;
    }

    boost::mpi::broadcast(world, RESAMPNESS   , 0);
    boost::mpi::broadcast(world, NUM_PARTICLES, 0);
    boost::mpi::broadcast(world, DIVIDER      , 0);
    boost::mpi::broadcast(world, curweight    , 0);
    boost::mpi::broadcast(world, weightfunc   , 0);

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    std::ifstream fin;
    
    ImageReaderType::Pointer reader;
    itk::ImageLinearIteratorWithIndex<Image4DType> iter;
    Image4DType::Pointer measInput;

    BoldModel model(zero_vector(BoldModel::SYSTEM_SIZE), weightfunc, curweight);

    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    
    /* Full Distribution */
    aux::DiracMixturePdf tmpX(BoldModel::SYSTEM_SIZE);
    if(rank == 0) {
        
        /* Open up the input */
        reader = ImageReaderType::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        cout << seriesfile << endl;;
        reader->SetFileName( seriesfile );
        reader->Update();
        measInput = reader->GetOutput();
        
        /* Create the iterator, to move forward in time for a particlular section */
        iter = itk::ImageLinearIteratorWithIndex<Image4DType>(measInput, 
                    measInput->GetRequestedRegion());
        iter.SetDirection(3);
        Image4DType::IndexType index = {{0, 0, 0, 0}};
        iter.SetIndex(index);
        
        /* Create a model */
        string str;
        itk::ExposeMetaData<double>(measInput->GetMetaDataDictionary(), 
                    "TemporalResolution", SAMPLETIME);
        cout << SAMPLETIME << endl;;
        if(SAMPLETIME == 0) {
            cout << "Image Had Invalid Temporal Resolution!" << endl;
            cout << "Using 2 seconds!" << endl;
            SAMPLETIME=2;
        }
        cout << left << setw(20) << "TR" << ": " << SAMPLETIME << endl;
    
        /* Generate Prior */
        if(!serialifile.empty()) {
            std::ifstream serialin(serialifile.c_str(), std::ios::binary);
            boost::archive::binary_iarchive inArchive(serialin);
            inArchive >> tmpX;
            if(NUM_PARTICLES != tmpX.getSize())
                cout << "Number of particles changed to " << tmpX.getSize();
            NUM_PARTICLES = tmpX.getSize();
        } else if(cheating) {
            model.generatePrior(tmpX, NUM_PARTICLES, cheat);
        } else {
            *out << "Generating prior" << endl;
            model.generatePrior(tmpX, NUM_PARTICLES);
//            aux::matrix tmp = tmpX.getDistributedCovariance();
//            *out << "Covariance: " << endl;
//            outputMatrix(*out, tmp);
//            *out << endl;
        }
    
    }
        
    boost::mpi::broadcast(world, tmpX, 0);

    /* Divide Initial Distribution among nodes */
    for(size_t i = rank ; i < NUM_PARTICLES ; i+= size) {
        x0.add(tmpX.get(i));
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


    Image4DType::Pointer measOutput, stateOutput, covOutput, partOutput;

    int meassize = 0;

    /////////////////////////////////////////////////////////////////////
    // output setup 
    /////////////////////////////////////////////////////////////////////
    if(rank == 0) {
        //BOLD
        measOutput = Image4DType::New();
        init4DImage(measOutput , measInput->GetRequestedRegion().GetSize()[0],
                1, 1, measInput->GetRequestedRegion().GetSize()[3]);
        measOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());

        //STATE
        stateOutput = Image4DType::New();
        init4DImage(stateOutput, 1, BoldModel::SYSTEM_SIZE, 1, 
                measInput->GetRequestedRegion().GetSize()[3]);
        stateOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());
        itk::EncapsulateMetaData<std::string>(stateOutput->GetMetaDataDictionary(),
                    "Dim0", "unused");
        itk::EncapsulateMetaData<std::string>(stateOutput->GetMetaDataDictionary(),
                    "Dim1", "systemmean");
        itk::EncapsulateMetaData<std::string>(stateOutput->GetMetaDataDictionary(),
                    "Dim2", "unused");
        itk::EncapsulateMetaData<std::string>(stateOutput->GetMetaDataDictionary(),
                    "Dim0", "time");

        //COVARIANCE
        covOutput = Image4DType::New();
        init4DImage(covOutput, 1, 
                BoldModel::SYSTEM_SIZE, BoldModel::SYSTEM_SIZE, 
                measInput->GetRequestedRegion().GetSize()[3]);
        covOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());
        itk::EncapsulateMetaData<std::string>(covOutput->GetMetaDataDictionary(),
                    "Dim0", "unused");
        itk::EncapsulateMetaData<std::string>(covOutput->GetMetaDataDictionary(),
                    "Dim1", "systemcov");
        itk::EncapsulateMetaData<std::string>(covOutput->GetMetaDataDictionary(),
                    "Dim2", "unusedcov");
        itk::EncapsulateMetaData<std::string>(covOutput->GetMetaDataDictionary(),
                    "Dim0", "time");
#ifdef PARTOUT
        //PARTICLES
        partOutput = Image4DType::New();
        init4DImage(partOutput, 1,  BoldModel::SYSTEM_SIZE, NUM_PARTICLES,
                measInput->GetRequestedRegion().GetSize()[3]*DIVIDER);
        partOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());
        itk::EncapsulateMetaData<std::string>(partOutput->GetMetaDataDictionary(),
                    "Dim0", "unused");
        itk::EncapsulateMetaData<std::string>(partOutput->GetMetaDataDictionary(),
                    "Dim1", "systemval");
        itk::EncapsulateMetaData<std::string>(partOutput->GetMetaDataDictionary(),
                    "Dim2", "particle");
        itk::EncapsulateMetaData<std::string>(partOutput->GetMetaDataDictionary(),
                    "Dim0", "time");
#endif //partout
        meassize = measInput->GetRequestedRegion().GetSize()[0];
    }

    boost::mpi::broadcast(world, SAMPLETIME, 0);

    /* Simulation Section */
    aux::DiracMixturePdf distr(BoldModel::SYSTEM_SIZE);
    aux::vector input(1);
    aux::vector meas(meassize);
    aux::vector mu(BoldModel::SYSTEM_SIZE);
    aux::symmetric_matrix cov(BoldModel::SYSTEM_SIZE);
    input[0] = 0;
    double nextinput;
    int disctime = 0;
    bool done = false;
    int tmp = 0;
    if(rank == 0 && !stimfile.empty()) {
        fin.open(stimfile.c_str());
        fin >> nextinput;
    }

    while(!done) {
#ifdef PARTOUT
        if( rank == 0 && !partfile.empty() ) {
            const std::vector<aux::DiracPdf>& particles = 
                        filter.getFilteredState().getAll();
            writeParticles(partOutput, particles, disctime);
        }
#endif //PARTOUT
        
        //TODO maybe this should be split up to prevent ranks from having
        //to move in lock-step
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
                cout << "Measuring at " <<  disctime/DIVIDER << endl;
                Image4DType::IndexType index = {{0, 0, 0, disctime/DIVIDER}};
                readVector(measInput, 0, meas, index);
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
            } else {
                cerr << "Total Weight: " << filter.getFilteredState().getTotalWeight() << endl;
                aux::vector weights = filter.getFilteredState().getWeights();
                outputVector(cerr, weights);
            }

            //time to resample
            if(ess < NUM_PARTICLES*RESAMPNESS) {
                *out << endl << " ESS: " << ess << ", Deterministic Resampling" << endl;
                filter.resample(&resampler);
                
                *out << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter.resample(&resampler_reg);
            } else
                *out << endl << " ESS: " << ess << ", No Resampling Necessary!" << endl;
        
            /* Get state */
            distr = filter.getFilteredState();
            mu = distr.getDistributedExpectation();
            cov = distr.getDistributedCovariance();
            if( rank == 0 ) {
                /* output measurement */
                
                //save states in an image
                Image4DType::IndexType index = {{0, 0, 0, disctime/DIVIDER}};
                writeVector(measOutput, 0, model.measure(mu), index);
                writeVector(stateOutput, 1, mu, index);
                writeMatrix(covOutput, 1, 2, cov, index);
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
        WriterType::Pointer writer = WriterType::New();
        writer->SetImageIO(itk::modNiftiImageIO::New());
        if(!serialofile.empty()) {
            std::ofstream serialout(serialofile.c_str(), std::ios::binary);
            boost::archive::binary_oarchive outArchive(serialout);
            outArchive << x0;
        }

        if(!boldfile.empty()) {
            writer->SetFileName(boldfile);  
            writer->SetInput(measOutput);
            writer->Update();
        }

        if(!statefile.empty()) {
            writer->SetFileName(statefile);  
            writer->SetInput(stateOutput);
            writer->Update();
        }
        
        if(!covfile.empty()) {
            writer->SetFileName(covfile);  
            writer->SetInput(covOutput);
            writer->Update();
        }
    }

  return 0;

}

