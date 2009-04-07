#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "BoldModel.hpp"

#include <indii/ml/aux/vector.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
    
using namespace std;

typedef itk::OrientedImage<double, 2> Image2DType;

int main (int argc, char** argv)
{
    srand(1333);
    if(argc != 8) {
        printf("Usage, %s <output> <out_timestep> <sim_timestep> <stoptime> ", argv[0]);
        printf("<numseries> <matlab_prefix> <stimfile>\n");
        printf("output is the image file that will contain the bold response data");
        printf("   time being the second direction\n");
        printf("out_timestep is the sample spacing (ex 2 seconds)\n");
        printf("sim_timestep is how often to sample for simulation, the smaller the better\n");
        printf("   (ex .001)\n");
        printf("stoptime what simulation time you want to stop (ex. 300.1 (seconds))\n");
        printf("numseries is the number of sections to simulate (ex. 5 )\n");
        printf("matlab_prefix is the prefix for the matlab files that will be generated:\n");
        printf("   <prefix>_state.out will contain a matrix of 7 + 4*<sections> by\n");
        printf("        <stoptime/out_timestep> with all the timeseries of state data in it\n");
        printf("   <prefix>_meas.out will contain a matrix of <sections> by\n");
        printf("        <stoptime/out_timestep> the bold responses in each section\n");
        printf("stimfile is very simple, a double time followed by the new value at that time\n");

        return -1;
    }

    //create a 2D output image of appropriate size.
    itk::ImageFileWriter< Image2DType >::Pointer writer = 
        itk::ImageFileWriter< Image2DType >::New();
    Image2DType::Pointer outputImage = Image2DType::New();

    Image2DType::RegionType out_region;
    Image2DType::IndexType out_index;
    Image2DType::SizeType out_size;

    const double stoptime = atof(argv[4]);
    const double outstep= atof(argv[2]);
    const double simstep = atof(argv[3]);
    const int series = atoi(argv[5]);
    const int endcount = (int)(stoptime/simstep);
    
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
    
    std::ifstream fin(argv[7]);
    std::ofstream fstate(std::string(argv[6]).append("state.out").c_str());
    std::ofstream fmeas(std::string(argv[6]).append("meas.out").c_str());
    
    fstate << "# Created by boldgen " << endl;
    fstate << "# name: statessim " << endl;
    fstate << "# type: matrix" << endl;
    fstate << "# rows: " << out_size[1] << endl;
    fstate << "# columns: " << BoldModel::SYSTEM_SIZE + 2 << endl;
    
    fmeas << "# Created by boldgen " << endl;
    fmeas << "# name: meassim " << endl;
    fmeas << "# type: matrix" << endl;
    fmeas << "# rows: " << out_size[1] << endl;
    fmeas << "# columns: " << 3 << endl;

    aux::vector system(BoldModel::SYSTEM_SIZE);
    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    model.generatePrior(x0, 10000);
    
//    system = x0.sample();
//    outputVector(std::cout, system);
//    std::cout << std::endl;

    int sample = 0;
    count = 0;
    double realt = 0;
    double prev = 0;

    //TODO implement multiple series (based on noise)
    aux::vector input(1);
    input[0] = 0;
    double nextinput;
    fin >> nextinput;
    for(count = 0 ; count  < endcount; count++) {
        //setup next timestep
        prev = realt;
        realt = count*simstep;
        if(!fin.eof() && realt >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
        }
        system = model.transition(system, realt, realt-prev, input);
        //TODO add noise to simulation
        
        //check to see if it is time to sample
        if(count == sample) {
            int i;
            
            //save states in a matlab file for comparison purposes
            fstate << realt << ' ' << input[0] << ' ';
            outputVector(fstate, system);
            fstate << endl;

            //TODO put multiple series here
            out_it.Value() = model.measure(system)[0];
            fmeas << realt << " " << input[0] << " " << model.measure(system)[0] << endl;
            for(i = 0 ; i < series ; i++) {
                ++out_it;
            }

            //move forward iterators
            assert(out_it.IsAtEndOfLine() && i == series);
            out_it.NextLine();
            sample += (int)(outstep/simstep);
        }
        
    }

    writer->SetFileName(argv[1]);  
    writer->SetInput(outputImage);
    writer->Update();
    return 0;
}

