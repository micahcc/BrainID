#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "BoldModel.hpp"

#include <indii/ml/aux/vector.hpp>

#include <boost/program_options.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <ctime>
    
using namespace std;
namespace opts = boost::program_options;

typedef itk::OrientedImage<double, 2> Image2DType;

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
            ("noisevar,v", opts::value<double>(), "Variance of Gaussian Noise to apply to bold signal")
            ("params,p", opts::value<string>(), "File with X0, Theta for simulation");

    opts::variables_map cli_vars;
    opts::store(opts::parse_command_line(argc, argv, desc), cli_vars);
    opts::notify(cli_vars);
    
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
        cout << "Reading Simulation Init/theta from: " 
                    << cli_vars["params"].as<string>() << endl;
    } else {
        cout << "Using random values for init/theta" << endl;
    }

    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    srand(1333);

    //create a 2D output image of appropriate size.
    itk::ImageFileWriter< Image2DType >::Pointer writer = 
        itk::ImageFileWriter< Image2DType >::New();
    Image2DType::Pointer outputImage = Image2DType::New();

    Image2DType::RegionType out_region;
    Image2DType::IndexType out_index;
    Image2DType::SizeType out_size;

    out_size[0] = series;
    //TODO deal with add error in double which could cause less or more
    //states to be simulated
    out_size[1] = (int)(stoptime/outstep)+1; //|T|T|T| + one for the series number
    
    out_index[0] = 0;
    out_index[1] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    outputImage->SetRegions( out_region );
    outputImage->Allocate();
    
    //setup iterator
    itk::ImageLinearIteratorWithIndex<Image2DType> 
                out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetDirection(0);

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
        fstate << "# rows: " << out_size[1] << endl;
        fstate << "# columns: " << BoldModel::SYSTEM_SIZE + 2 << endl;
    }

    if(fmeas.is_open()) {
        fmeas << "# Created by boldgen " << endl;
        fmeas << "# name: meassim " << endl;
        fmeas << "# type: matrix" << endl;
        fmeas << "# rows: " << out_size[1] << endl;
        fmeas << "# columns: " << 3 << endl;
    }
    
    //Used to add noise
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (int)(time(NULL)*rank)/11.);

    aux::vector system(BoldModel::SYSTEM_SIZE);
    if(cli_vars.count("params") == 0) {
        aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
        model.generatePrior(x0, 10000);
        system = x0.sample();
    } else {
        ifstream init(cli_vars["params"].as<string>().c_str());
        for(int i = 0 ; i < BoldModel::SYSTEM_SIZE ; i++) {
            init >> system[i];
        }
        if(init.eof()) {
            cerr << "Error not enough arguments given" << endl;
            outputVector(std::cout, system);
            exit(-1);
        }
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
    for(count = 0 ; count  < endcount; count++) {
        //setup next timestep
        prev = realt;
        realt = count*simstep;
        if(fin.is_open() && !fin.eof() && realt >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
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

