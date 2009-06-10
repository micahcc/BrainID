#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"
#include "BoldModel.hpp"

#include <indii/ml/aux/vector.hpp>

#include <boost/program_options.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <sstream>
#include <cmath>

using namespace std;
namespace opts = boost::program_options;

const int SERIES_DIR = 0;
const int PARAM_DIR = 1;
const int TIME_DIR = 3;

typedef itk::OrientedImage<double, 4> Image4DType;

double stoptime;
double outstep;
double simstep;
int series;
int endcount;

ifstream fin;

string boldfile;
string statefile;

double noise_snr;

typedef struct {
    ofstream fout;
    double p;
    double t;
    string filename;
} stimproc;

stimproc stim;

aux::vector systemstate(BoldModel::SYSTEM_SIZE);

void parse_cli(int argc, char** argv, BoldModel& model) 
{
    //CLI
    opts::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("out,o", opts::value<string>(), "image file to write to")
            ("outtime,t", opts::value<double>(), "How often to sample")
            ("simtime,s", opts::value<double>(), "Step size for sim, smaller is more accurate")
            ("endtime,e", opts::value<double>(), "What time to end at")
            ("numseries,n", opts::value<int>(), "Number of brain regions to simulate")
            ("statefile,x", opts::value<string>(), "file to write out state data to")
            ("inputstim,i", opts::value<string>(), "file to read in stimuli from")
            ("randstim,r", opts::value<string>(), "create a random stimulus then write to"
                        "file=<where to write to>,t=<time between changes,p=<probability of 1>")
            ("noisesnr,v", opts::value<double>(), "SNR of Gaussian Noise to apply to bold signal")
            ("file,f", opts::value<string>(), "File with X0, Theta for simulation")
            ("params,p", opts::value<string>(), "Parameters to pass "
                        "into the simulation, enclosed in quotes \"Tau_s Tau_f Epsilon Tau_0 "
                        "alpha E_0 V_0 v_t0 q_t0 s_t0 f_t0\"");
    
    opts::variables_map cli_vars;
    try {
        opts::store(opts::parse_command_line(argc, argv, desc), cli_vars);
        opts::notify(cli_vars);
    } catch(...) {
        cout << "Error! Improper Command Line Option Given!" << endl << endl;
        cout << desc << endl;
        exit(-6);
    }
    
    if(cli_vars.count("help")) {
        cout << desc << endl;
        exit(0);
    }
    
    if(cli_vars.count("out")) {
        cout << left << setw(20) << "Output Image" << ":" 
                    << cli_vars["out"].as<string>() << endl;
        boldfile = cli_vars["out"].as<string>();
    } else {
        cout << left << setw(20) << "Warning" << ":Not outputing the simulated "
                    "timeseries image because no name was given" << endl;
    }
    
    if(cli_vars.count("outtime")) {
        cout << setw(20) << "Out timestep" << ":" 
                    << cli_vars["outtime"].as<double>() << endl;
        outstep = cli_vars["outtime"].as<double>();
    } else {
        cout << setw(20) << "Error! " << "Must give an output timestep!" << endl;
        exit(-1);
    }
    
    if(cli_vars.count("simtime")) {
        cout << left << setw(20) << "Simulation timestep" << ":" 
                    << cli_vars["simtime"].as<double>() << endl;
        simstep = cli_vars["simtime"].as<double>();
    } else {
        cout << setw(20) << "Error!" << "Must give a simulation timestep!" << endl;
        exit(-2);
    }
    
    if(cli_vars.count("statefile")) {
        cout << left << setw(20) << "Statefile" << ":" 
                    << cli_vars["statefile"].as<string>() << endl;
        statefile = cli_vars["statefile"].as<string>();
    } else {
        cout << left << setw(20) << "Warning" << ":Not outputing any state information" 
                    << endl;
    }
    
    if(cli_vars.count("endtime")) {
        cout << setw(20) << left << "Ending at time" << ":" 
                    << cli_vars["endtime"].as<double>() << endl;
        stoptime = cli_vars["endtime"].as<double>();
        endcount = (int)(stoptime/simstep);
    } else {
        cout << "Must give a time to end the simulation!" << endl;
        exit(-3);
    }
    
    if(cli_vars.count("numseries")) {
        cout << "WARNING numseries is not implemented yet!" << endl;
        series = cli_vars["numseries"].as<int>();
    } else {
        series = 1;
    }
    cout << left << setw(20) << "Number of series" << ":" 
                << series << endl;
    
    if(cli_vars.count("inputstim")) {
        string tmp = cli_vars["inputstim"].as<string>();
        cout << setw(20) << left << "Stimuli File" << ": " << tmp << endl;
        fin.open(tmp.c_str());
    } else if(cli_vars.count("randstim")) {
        string tmp;
        string tmpopts = cli_vars["randstim"].as<string>();
        tmpopts.append(",");
        size_t index;
        
        while((index = tmpopts.find(',')) != string::npos) {
            tmp = tmpopts.substr(0,index);
            if(!tmp.compare(0,5,"file=")) {
                istringstream iss2(tmp.substr(5));
                iss2 >> stim.filename;
            } else if(!tmp.compare(0,2,"t=")) {
                istringstream iss2(tmp.substr(2));
                iss2 >> stim.t;
            } else if(!tmp.compare(0,2,"p=")) {
                istringstream iss2(tmp.substr(2));
                iss2 >> stim.p;
            }
            tmpopts.erase(0,index+1);
        }
        stim.fout.open(stim.filename.c_str());
        cout << "Creating binomial process with p=" << stim.p 
                    << " t=" << stim.t << " filename="
                    << stim.filename << endl;
    } else {
        cout << "No stimuli given, will decay freely" << endl;
    }
    
    if(cli_vars.count("noisesnr")) {
        noise_snr = cli_vars["noisesnr"].as<double>();
        cout << "Using variance of: " << noise_snr << endl;
    } else {
        noise_snr = 0;
    }
    
    if(cli_vars.count("params")) {
        cout << "Reading Simulation Init/theta from command line: "  << endl
                    << cli_vars["params"].as<string>() << endl;

        istringstream iss(cli_vars["params"].as<string>());
        
        for(int i = 0 ; i < BoldModel::SYSTEM_SIZE ; i++) {
            if(iss.eof()) {
                cerr << "Error not enough arguments given on command line" << endl;
                exit(-3);
            }
            iss >> systemstate[i];
        }
        
    } else if(cli_vars.count("file")) {
        cout << "Reading Simulation Init/theta from: " 
                    << cli_vars["file"].as<string>() << endl;
        
        ifstream init(cli_vars["file"].as<string>().c_str());
        
        for(int i = 0 ; i < BoldModel::SYSTEM_SIZE ; i++)
            init >> systemstate[i];

        if(init.eof()) {
            cerr << "Error not enough arguments given in file" << endl;
            outputVector(std::cout, systemstate);
            cerr << endl;
            exit(-1);
        }
    } else {
        cout << "Using random values for init/theta" << endl;
        aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);

        aux::vector mean(BoldModel::SYSTEM_SIZE);
        aux::symmetric_matrix cov = aux::zero_matrix(BoldModel::SYSTEM_SIZE);
        
        //set the variances for all the variables
        cov(0,0) = 1.07*1.07;
        cov(1,1) = 1.51*1.51;
        cov(2,2) = 0.014*.014;
        cov(3,3) = 1.5*1.5;
        cov(4,4) = .004*.004;
        cov(5,5) = .072*.072;
        cov(6,6) = .6e-2*.6e-2;

        cov(7,7) = .1;
        cov(8,8) = .1;
        cov(9,9) = .1;
        cov(10,10) = .1;

        model.generatePrior(x0, 10000, cov);
        systemstate = x0.sample();

#ifdef ZEROSTART
        //this assumes you start at resting state, which is not a bad
        //assumption. Plus this way you don't get negative bold.
        for(int i = BoldModel::THETA_SIZE ; i < BoldModel::SYSTEM_SIZE ; i++) {
            if((i-BoldModel::THETA_SIZE)%BoldModel::STATE_SIZE == 2) 
                systemstate[i] = 1;
            else
                systemstate[i] = 0;
        }
#endif //ZEROSTART
    } 

    std::cout << std::endl;
    outputVector(std::cout, systemstate);
    std::cout << std::endl;

}

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

//dir1 should be the direction of several separate series
//dir2 should be the direction that you want to get rms of
void get_rms(Image4DType::Pointer in, size_t dir1, size_t dir2, 
            vector<double>& out)
{
    out.assign(in->GetRequestedRegion().GetSize()[dir1], 0);
    
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                iter(in, in->GetRequestedRegion());
    iter.SetFirstDirection(dir1);
    iter.SetSecondDirection(dir2);
    iter.GoToBegin();

    //sum of squares
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        while(!iter.IsAtEndOfLine()) {
            out[ii] += pow(iter.Get(),2);
            ii++;
            ++iter;
        }
        iter.NextLine();
    }

    //root mean
    for(size_t i = 0 ; i < out.size() ; i++) {
        out[i] = sqrt(out[i]/in->GetRequestedRegion().GetSize()[dir2]);
    }
}

void add_noise(Image4DType::Pointer in, double snr, gsl_rng* rng) 
{
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                iter(in, in->GetRequestedRegion());
    iter.SetFirstDirection(SERIES_DIR);
    iter.SetSecondDirection(TIME_DIR);
    iter.GoToBegin();
    
    vector<double> rms(series, 0);

    //calculate rms of image over time
    get_rms(in, SERIES_DIR, TIME_DIR, rms);
    for(size_t ii= 0 ; ii<rms.size() ; ii++) {
        cout << "RMS: " << ii << ":" << rms[ii] << endl;
    }

    iter.GoToBegin();
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        while(!iter.IsAtEndOfLine()) {
            iter.Value() += gsl_ran_gaussian(rng, rms[ii]/sqrt(snr));
            ++iter;
            ii++;
        }
        iter.NextLine();
    }
    
    get_rms(in, SERIES_DIR, TIME_DIR, rms);
    for(size_t ii= 0 ; ii<rms.size() ; ii++) {
        cout << "RMS: " << ii << " " << rms[ii] << endl;
    }
}

int main (int argc, char** argv)
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    BoldModel model;
    parse_cli(argc, argv, model);

    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    srand(1333);

    //create a 4D output image of appropriate size.
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
        itk::ImageFileWriter< Image4DType >::New();
    Image4DType::Pointer measImage = Image4DType::New();
    
    //TODO deal with add error in double which could cause less or more
    //states to be simulated
    //tlen = |T|T|T| + one for the series number
    init4DImage(measImage, series, 1, 1, (int)ceil(stoptime/outstep));
  
    itk::MetaDataDictionary dict = measImage->GetMetaDataDictionary();
    itk::EncapsulateMetaData<double>(dict, "TemporalResolution", 2.0);
    itk::EncapsulateMetaData<unsigned int>(dict, "NumSections", series);
    itk::EncapsulateMetaData<unsigned int>(dict, "TimeDim", 3);
    itk::EncapsulateMetaData<unsigned int>(dict, "SectionDim", 0);
    //fill in mapping of Section Index to number section number i+5 
    for(int i=0 ; i<series ; i++ ) {
        ostringstream oss;
        oss.str("");
        oss << "MapIndex " << i;
        itk::EncapsulateMetaData<unsigned int>(dict, oss.str(), i+5);
    }
    measImage->SetMetaDataDictionary(dict);

    //initialize first line in t direction to hold the section number
    
    Image4DType::Pointer outState = Image4DType::New();
    init4DImage(outState, series, BoldModel::SYSTEM_SIZE, 1, 
                (int)ceil(stoptime/outstep));

    itk::EncapsulateMetaData<unsigned int>(dict, "StateDim", 1);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexTauS",    0);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexTauF",    1);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexEpsilon", 2);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexTau0",    3);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexAlpha",   4);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexE0",      5);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexV0",      6);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexVT",      7);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexQT",      8);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexST",      9);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexFT",     10);
#define TEST
#ifdef TEST
    itk::EncapsulateMetaData<string>(dict, "0010|0040", "M");
#endif //TEST
 
    outState->SetMetaDataDictionary(dict);

//#ifdef TEST
//    unsigned int tmp1000;
//    dict = outState->GetMetaDataDictionary();
//    itk::ExposeMetaData<unsigned int>(dict, "IndexE0", tmp1000);
//    cout << "IndexE0 " << tmp1000 << endl;;
//    dict.Print(cout);
//#endif

    //setup iterators
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                meas_it(measImage, measImage->GetRequestedRegion());
    meas_it.SetFirstDirection(SERIES_DIR);
    meas_it.SetSecondDirection(TIME_DIR);
    meas_it.GoToBegin();
    
    itk::ImageLinearIteratorWithIndex<Image4DType> 
                state_it(outState, outState->GetRequestedRegion());
    state_it.SetDirection(PARAM_DIR);

    //Used to add noise
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (int)((time(NULL)*rank)/11.));

//    outputVector(std::cout, systemstate);
//    std::cout << std::endl;

    double sample = 0;
    int count = 0;
    double realt = 0;
    double prev = 0;

    //TODO implement multiple series (based on noise)
    aux::vector input(1);
    input[0] = 0;
    double nextinput;
    if(fin.is_open())
        fin >> nextinput;
    else if(stim.fout.is_open())  {
        input[0] = (double)rand()/RAND_MAX < stim.p;
        stim.fout << 0 << " " << input[0] << endl;
        nextinput = stim.t;
    }

//    cout << endcount << endl;
    for(count = 0 ; count  < endcount; count++) {
        //setup next timestep
        prev = realt;
        realt = count*simstep;
        if(fin.is_open() && !fin.eof() && realt >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
        } else if(stim.fout.is_open() && realt >= nextinput) {
            input[0] = (double)rand()/RAND_MAX < stim.p;
            stim.fout << nextinput << " " << input[0] << endl;
            nextinput += stim.t;
        }

        model.transition(systemstate, realt, realt-prev, input);
        //TODO add noise to simulation
        
        //check to see if it is time to sample
        if(count == (int)sample) {
            while(!meas_it.IsAtEndOfLine()) {
                
                //save states in an image
                state_it.SetIndex(meas_it.GetIndex());
                
                int j = 0;
                while(!state_it.IsAtEndOfLine()){
                    state_it.Set(systemstate[j++]);
                    ++state_it;
                }
                
                meas_it.Value() = model.measure(systemstate)[0];
                ++meas_it;
            }

            meas_it.NextLine();
            
            //TODO should use an absolute value here to prevent error
            sample += (outstep/simstep);
        }
        
    }

    if(noise_snr != 0)
        add_noise(measImage, noise_snr, rng);
    
    if(!boldfile.empty()) {
        writer->SetFileName(boldfile);  
        writer->SetInput(measImage);
        writer->Update();
    }
    
    if(!statefile.empty()) {
        writer->SetFileName(statefile);  
        writer->SetInput(outState);
        writer->Update();
    }
  
    gsl_rng_free(rng);
    return 0;
}

