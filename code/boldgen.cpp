#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
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

using namespace std;
namespace opts = boost::program_options;

typedef itk::OrientedImage<double, 4> Image4DType;

int main (int argc, char** argv)
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    double stoptime;
    double outstep;
    double simstep;
    int series;
    int endcount;
    
    ifstream fin;
    ofstream fstate;
    ofstream fmeas;
    
    string imagename;

    double noise_var;
    
    struct {
        ofstream fout;
        double p;
        double t;
        string filename;
    } stimproc;
    
    aux::vector system(BoldModel::SYSTEM_SIZE);
    
    //CLI
    opts::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("out,o", opts::value<string>(), "image file to write to")
            ("outtime,t", opts::value<double>(), "How often to sample")
            ("simtime,s", opts::value<double>(), "Step size for sim, smaller is more accurate")
            ("endtime,e", opts::value<double>(), "What time to end at")
            ("numseries,n", opts::value<int>(), "Number of brain regions to simulate")
            ("matlab,m", opts::value<string>(), "prefix for matlab files")
            ("inputstim,i", opts::value<string>(), "file to read in stimuli from")
            ("randstim,r", opts::value<string>(), "create a random stimulus then write to"
                        "file=<where to write to>,t=<time between changes,p=<probability of 1>")
            ("noisevar,v", opts::value<double>(), "Variance of Gaussian Noise to apply to bold signal")
            ("file,f", opts::value<string>(), "File with X0, Theta for simulation")
            ("params,p", opts::value<string>(), "Parameters to pass "
                        "into the simulation, enclosed in quotes \"Tau_s Tau_f Epsilon Tau_0 "
                        "alpha E_0 V_0 v_t0 q_t0 s_t0 f_t0\"");

    opts::variables_map cli_vars;
    try {
        opts::store(opts::parse_command_line(argc, argv, desc), cli_vars);
        opts::notify(cli_vars);
    } catch(...) {
        cout << "Improper Command Line Option Given!" << endl << endl;
        cout << desc << endl;
        return -6;
    }
    
    if(cli_vars.count("help")) {
        cout << desc << endl;
        return 1;
    }
    
    if(cli_vars.count("out")) {
        cout << "Output Image: " << cli_vars["out"].as<string>() << endl;
        imagename = cli_vars["out"].as<string>();
    } else {
        cout << "Not outputing the simulated timeseries image because no name was given" << endl;
    }
    
    if(cli_vars.count("outtime")) {
        cout << "Out timestep: " << cli_vars["outtime"].as<double>() << endl;
        outstep = cli_vars["outtime"].as<double>();
    } else {
        cout << "Must give an output timestep!" << endl;
        return -1;
    }
    
    if(cli_vars.count("simtime")) {
        cout << "Simulation timestep: " << cli_vars["simtime"].as<double>() << endl;
        simstep = cli_vars["simtime"].as<double>();
    } else {
        cout << "Must give a simulation timestep!" << endl;
        return -2;
    }
    
    if(cli_vars.count("endtime")) {
        cout << "Ending at time: " << cli_vars["endtime"].as<double>() << endl;
        stoptime = cli_vars["endtime"].as<double>();
        endcount = (int)(stoptime/simstep);
    } else {
        cout << "Must give a time to end the simulation!" << endl;
        return -3;
    }
    
    if(cli_vars.count("numseries")) {
        cout << "Number of series: " << cli_vars["numseries"].as<int>() << endl;
        cout << "WARNING this is not implemented yet!" << endl;
        series = cli_vars["numseries"].as<int>();
    } else {
        cout << "Simulation 1 section" << endl;
        series = 1;
    }
    
    if(cli_vars.count("matlab")) {
        string tmp = cli_vars["matlab"].as<string>();
        string tmp2  = tmp;
        cout << "Outputing to matlab files: " << tmp << "state.out and " << tmp
                    << "meas.out" << endl;
        fstate.open(tmp.append("state.out").c_str());
        fmeas.open(tmp2.append("meas.out").c_str());
    } else {
        cout << "No matlab output will be generated" << endl;
    }
    
    if(cli_vars.count("inputstim")) {
        string tmp = cli_vars["inputstim"].as<string>();
        cout << "Will read stimuli from: " << tmp << endl;
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
                iss2 >> stimproc.filename;
            } else if(!tmp.compare(0,2,"t=")) {
                istringstream iss2(tmp.substr(2));
                iss2 >> stimproc.t;
            } else if(!tmp.compare(0,2,"p=")) {
                istringstream iss2(tmp.substr(2));
                iss2 >> stimproc.p;
            }
            tmpopts.erase(0,index+1);
        }
        stimproc.fout.open(stimproc.filename.c_str());
        cout << "Creating binomial process with p=" << stimproc.p 
                    << " t=" << stimproc.t << " filename="
                    << stimproc.filename << endl;
    } else {
        cout << "No stimuli given, will decay freely" << endl;
    }
    
    if(cli_vars.count("noisevar")) {
        noise_var = cli_vars["noisevar"].as<double>();
        cout << "Using variance of: " << noise_var << endl;
    } else {
        noise_var = 0;
    }
    
    if(cli_vars.count("params")) {
        cout << "Reading Simulation Init/theta from command line: " 
                    << cli_vars["params"].as<string>() << endl;

        istringstream iss(cli_vars["params"].as<string>());
        
        for(int i = 0 ; i < BoldModel::SYSTEM_SIZE ; i++) {
            if(iss.eof()) {
                cerr << "Error not enough arguments given on command line" << endl;
                exit(-3);
            }
            iss >> system[i];
        }
        
    } else if(cli_vars.count("file")) {
        cout << "Reading Simulation Init/theta from: " 
                    << cli_vars["file"].as<string>() << endl;
        
        ifstream init(cli_vars["file"].as<string>().c_str());
        
        for(int i = 0 ; i < BoldModel::SYSTEM_SIZE ; i++)
            init >> system[i];

        if(init.eof()) {
            cerr << "Error not enough arguments given in file" << endl;
            outputVector(std::cout, system);
            cerr << endl;
            exit(-1);
        }
    } else {
        cout << "Using random values for init/theta" << endl;
    }
    

    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    srand(1333);

    //create a 2D output image of appropriate size.
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
        itk::ImageFileWriter< Image4DType >::New();
    Image4DType::Pointer outputImage = Image4DType::New();

    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index;
    Image4DType::SizeType out_size;

    out_size[0] = series;
    out_size[1] = 1;
    out_size[2] = 1;
    //TODO deal with add error in double which could cause less or more
    //states to be simulated
    out_size[3] = (int)(stoptime/outstep)+1; //|T|T|T| + one for the series number
    
    out_index[0] = 0;
    out_index[1] = 0;
    out_index[2] = 0;
    out_index[3] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    
    //setup iterator
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(3);

    int count = 0;
    while(!out_it.IsAtEndOfLine()) {
        out_it.Value() = count++;
        ++out_it;
    }
    out_it.NextLine();

    BoldModel model;
    
    if(fstate.is_open()) {
        fstate << "# Created by boldgen " << endl;
        fstate << "# name: statessim " << endl;
        fstate << "# type: matrix" << endl;
        fstate << "# rows: " << out_size[3] -1 << endl;
        fstate << "# columns: " << BoldModel::SYSTEM_SIZE + 1 << endl;
    }

    if(fmeas.is_open()) {
        fmeas << "# Created by boldgen " << endl;
        fmeas << "# name: meassim " << endl;
        fmeas << "# type: matrix" << endl;
        fmeas << "# rows: " << out_size[3] - 1<< endl;
        fmeas << "# columns: " << 3 << endl;
    }
    
    //Used to add noise
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (int)((time(NULL)*rank)/11.));

    if(cli_vars.count("file") == 0 && cli_vars.count("params") == 0) {
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
        system = x0.sample();

#ifdef ZEROSTART
        //this assumes you start at resting state, which is not a bad
        //assumption. Plus this way you don't get negative bold.
        for(int i = BoldModel::THETA_SIZE ; i < BoldModel::SYSTEM_SIZE ; i++) {
            if((i-BoldModel::THETA_SIZE)%BoldModel::STATE_SIZE == 2) 
                system[i] = 1;
            else
                system[i] = 0;
        }
#endif //ZEROSTART
    } 

    outputVector(std::cout, system);
    std::cout << std::endl;

    int sample = 0;
    count = 0;
    double realt = 0;
    double prev = 0;

    //TODO implement multiple series (based on noise)
    aux::vector input(1);
    input[0] = 0;
    double nextinput;
    if(fin.is_open())
        fin >> nextinput;
    else if(stimproc.fout.is_open())  {
        input[0] = (double)rand()/RAND_MAX < stimproc.p;
        stimproc.fout << 0 << " " << input[0] << endl;
        nextinput = stimproc.t;
    }
    for(count = 0 ; count  < endcount; count++) {
        //setup next timestep
        prev = realt;
        realt = count*simstep;
        if(fin.is_open() && !fin.eof() && realt >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
        } else if(stimproc.fout.is_open() && realt >= nextinput) {
            input[0] = (double)rand()/RAND_MAX < stimproc.p;
            stimproc.fout << nextinput << " " << input[0] << endl;
            nextinput += stimproc.t;
        }

        model.transition(system, realt, realt-prev, input);
        //TODO add noise to simulation
        
        //check to see if it is time to sample
        if(count == sample) {
            int i;
        
            //save states in a matlab file for comparison purposes
            if(fstate.is_open()){
                fstate << setw(10) << realt;
                outputVector(fstate, system);
                fstate << endl;
            }

            if(fmeas.is_open()) {
                fmeas << setw(10) << realt << setw(10) << input[0] << setw(14) 
                            << model.measure(system)[0] << endl;
            }
            
            //TODO put multiple series here
            out_it.Value() = model.measure(system)[0] + 
                        gsl_ran_gaussian(rng, sqrt(noise_var));
            for(i = 0 ; i < series ; i++) {
                ++out_it;
            }

            //move forward iterators
            assert(out_it.IsAtEndOfLine() && i == series);
            out_it.NextLine();

            //TODO should use an absolute value here to prevent error
            sample += (int)(outstep/simstep);
        }
        
    }

    if(!imagename.empty()) {
        writer->SetFileName(imagename);  
        writer->SetInput(outputImage);
        writer->Update();
    }
    
    gsl_rng_free(rng);
    return 0;
}

