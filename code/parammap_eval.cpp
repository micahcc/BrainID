//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan
#include "version.h"

#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMetaDataObject.h>

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include "tools.h"
#include "modNiftiImageIO.h"

#include <cmath>
#include <iostream>
#include <string>

#include <vul/vul_arg.h>

using namespace std;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;

namespace aux = indii::ml::aux;

double readout(State& state, const std::vector<double>& params)
{
    return params[V_0]*(params[A_1]*(1-state.Q)-params[A_2]*(1-state.V));
}

int transition(State& state, const std::vector<double>& params, double delta, 
            double in)
{
    State change;
    change.S = params[EPSILON]*in - state.S/params[TAU_S] - 
                (state.F - 1.)/params[TAU_F];
    change.F = state.S;
    change.V = (state.F - pow(state.V, 1./params[ALPHA]))/params[TAU_0];
    change.Q = (state.F*(1.-pow(1.-params[E_0],1./state.F))/params[E_0] -
                state.Q/pow(state.V, 1.-1./params[ALPHA]))/params[TAU_0];
    state.S += change.S*delta;
    state.F += change.F*delta;
    state.V += change.V*delta;
    state.Q += change.Q*delta;
    if(isinf(state.S) || isnan(state.S)) return -1;
    if(isinf(state.F) || isnan(state.F)) return -2;
    if(isinf(state.V) || isnan(state.V)) return -3;
    if(isinf(state.Q) || isnan(state.Q)) return -4;
    return 0;
}

void filllist(std::list< double >& output, Image4DType* input,
            Image4DType::IndexType pos)
{       
    ImgIter iter(input, input->GetRequestedRegion());
    iter.SetDirection(3);
    iter.SetIndex(pos);

    while(!iter.IsAtEndOfLine()) {
        output.push_back(iter.Get());
        ++iter;
    }
}

void fillvector(std::vector<double>& output, Image4DType* input,
            Image4DType::IndexType pos)
{       
    ImgIter iter(input, input->GetRequestedRegion());
    iter.SetDirection(3);
    iter.SetIndex(pos);

    output.resize(input->GetRequestedRegion().GetSize()[3]);
    int i = 0;
    while(!iter.IsAtEndOfLine()) {
        output[i] = iter.Get();
        ++iter;
        i++;
    }
}

double mse(const std::list<double> listA, const std::list<double> listB) 
{
    std::list<double>::const_iterator itA = listA.begin();
    std::list<double>::const_iterator itB = listB.begin();
    double acc = 0;
    while(itA != listA.end() && itB != listB.end()) {
        acc += *itA + *itB;
        itA++; itB++;
    }
    acc /= listA.size();
    return acc;
}
                
void simulate(std::list<double>& sim, const std::vector<double>& params,
            double dt_s, double dt_l, const std::vector<Activation>& input_vector,
            unsigned int length)
{
    //Set up simulation state, initialize
    State state;
    state.V = 1;
    state.Q = 1;
    state.F = 1;
    state.S = 0;

    //Counters/temps
    double input = 0;
    size_t sample_index = 0;
    size_t input_index = 0;
    double rt = 0;
    double prev_rt = 0;

    for(size_t ii = 0 ; sim.size() != length; ii++) {
        prev_rt = rt;
        rt = ii*dt_s; //"real" time

        /*Update Input If there is a change for the current time*/
        while(input_index < input_vector.size() && 
                    input_vector[input_index].time <= rt) {
            input = input_vector[input_index].level;
            input_index++;
        }

        /*Update state variables*/
        int res = transition(state, params, rt-prev_rt, input);
        switch(res) {
            case -1:
                printf("Error in s at %f - %f\n", rt, prev_rt);
                printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
                printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                        "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                        params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                        params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
                exit(res);
            case -2:
                printf("Error in f at %f - %f\n", rt, prev_rt);
                printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
                printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                        "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                        params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                        params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
                exit(res);
            case -3:
                printf("Error in v at %f - %f\n", rt, prev_rt);
                printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
                printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                        "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                        params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                        params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
                exit(res);
            case -4:
                printf("Error in q at %f - %f\n", rt, prev_rt);
                printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
                printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                        "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                        params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                        params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
                exit(res);
        }

        /*If it is time to sample, do so*/
        while(dt_l*sample_index <= rt) {
            sim.push_back(readout(state, params));
            sample_index++;
        }
    }
    
}
            
/* Main Function */
int main(int argc, char* argv[])
{
    vul_arg<string> a_params(0, "4D param file, in the order: TAU_0, ALPHA,"
                "E_0, V_0, TAU_S, TAU_F, EPSILON");
    vul_arg<string> a_input(0, "4D Timeseries file to compare simulation with.");
    vul_arg<string> a_output(0, "3D Output Image with MSE");
    
    vul_arg<unsigned> a_divider("-d", "Intermediate Steps between samples.", 128);
    vul_arg<string> a_stimfile("-s", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<double> a_timestep("-t", "TR (timesteps in 4th dimension)", 2);
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("No Warning, just echoing");

    Image4DType::Pointer compImage;
    Image4DType::Pointer paramImage;
    Image3DType::Pointer output;

    std::vector<Activation> input;

    /* Open up the input */
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_params() );
    reader->Update();
    paramImage = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_params().c_str());
        exit(-1);
    }

    try{
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_input() );
    reader->Update();
    compImage = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_input().c_str());
        exit(-2);
    }
    
    unsigned int xlen = compImage->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = compImage->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = compImage->GetRequestedRegion().GetSize()[2];
    
    /* Open Stimulus file */
    if(!a_stimfile().empty()) {
        input = read_activations(a_stimfile().c_str());
        if(input.empty()) 
            return -1;
    }

    if(input.size() == 0) {
        Activation tmp;
        tmp.time = 0;
        tmp.level= 0;
        input = std::vector<Activation>(1,tmp);
    }

    /* Create Output Images */
    cout << "Creating Output Image" << endl;
    Image3DType::SizeType size3 = {{xlen, ylen, zlen}};
    
    output = Image3DType::New();
    output->SetRegions(size3);
    output->Allocate();
    output->FillBuffer(0);

    for(unsigned int xx = 0 ; xx < xlen ; xx++) {
        for(unsigned int yy = 0 ; yy < ylen ; yy++) {
            for(unsigned int zz = 0 ; zz < zlen ; zz++) {
                //initialize some variables
                Image4DType::IndexType index4 = {{xx, yy, zz, 0}};
                Image3DType::IndexType index3 = {{xx, yy, zz}};
                vector<double> params;
                fillvector(params, paramImage, index4);
                list<double> sim;
                simulate(sim, params, 1./a_divider(), a_timestep(),input, 
                            compImage->GetRequestedRegion().GetSize()[3]);
                
                list<double> comparison;
                filllist(comparison, compImage, index4);
                output->SetPixel(index3, mse(comparison, sim));
            }
        }
    }
    
    //write final output
    {
    itk::ImageFileWriter<Image3DType>::Pointer out = 
        itk::ImageFileWriter<Image3DType>::New();
    out->SetInput(output);
    out->SetFileName(a_output());
    cout << "Writing: " << a_output() << endl;
    out->Update();
    }

    return 0;

}



