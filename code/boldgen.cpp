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
    if(argc != 5) {
        printf("Usage, %s <output> <out_timestep> <sim_timestep> <stoptime> <numseries>", argv[0]);
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

