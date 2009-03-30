#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "BoldModel.hpp"

#include <indii/ml/aux/vector.hpp>

#include <stdio.h>
#include <stdlib.h>

typedef itk::OrientedImage<double, 2> Image2DType;

int main (int argc, char** argv)
{
    if(argc != 7) {
        printf("Usage, %s <output> <out_timestep> <sim_timestep> <stoptime> ", argv[0]);
        printf("<numseries> <matlab_prefix>\n");
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

        return -1;
    }

    //create a 2D output image of appropriate size.
    itk::ImageFileWriter< Image2DType >::Pointer writer = 
        itk::ImageFileWriter< Image2DType >::New();
    Image2DType::Pointer outputImage = Image2DType::New();

    Image2DType::RegionType out_region;
    Image2DType::IndexType out_index;
    Image2DType::SizeType out_size;

    double stoptime = atof(argv[4]);
    double outstep= atof(argv[2]);
    double simstep = atof(argv[3]);
    int series = atoi(argv[5]);
    
    out_size[0] = series;
    out_size[1] = (int)(stoptime/outstep)+1+1; //|T|T|T| + one for the series number
    
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

    aux::vector systems[series];
    aux::DiracMixturePdf x0(BoldModel::SYSTEM_SIZE);
    model.generatePrior(x0, 10000);
    
    for(int i=0 ; i<series ; i++) {
        systems[i] = x0.sample();
        std::cout << i << "\t";
        outputVector(std::cout, systems[i]);
        std::cout << std::endl;
    }

    double sample = 0;
    double t = 0;
    aux::zero_vector input(1);
    while (t < stoptime) {
        //for now
        if(t > sample) {
            sample += outstep;
            int i;
            for(i = 0 ; i<series && !out_it.IsAtEndOfLine() ; i++) {
                out_it.Value() = model.measure(systems[i])[0];
                ++out_it;
            }
            assert(out_it.IsAtEndOfLine() && i == series);
            out_it.NextLine();
        }
        
        //for next time
        t += simstep;
        for(int i = 0 ; i < series ; i++) {
            systems[i] = model.transition(systems[i], t, simstep, input);
        }
    }

    writer->SetFileName(argv[1]);  
    writer->SetInput(outputImage);
    writer->Update();
    return 0;
}

