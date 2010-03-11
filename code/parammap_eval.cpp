//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan
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
    vul_arg<string> a_pEst(0, "4/5D param file, should be smaller (first 3 Dims) image, "
                "with the 4th dimension as time and the 5th dimension in the order "
                "TAU_0, ALPHA, E_0, V_0, TAU_S, TAU_F, EPSILON");
    vul_arg<string> a_pTrue(0, "True parameters, Either: \n\t\t4D param file, in the "
                "order: TAU_0, ALPHA, E_0, V_0, TAU_S, TAU_F, EPSILON, "
                "OR, \n\t\ttext file with parameters in the same order, one set");
    vul_arg<string> a_output(0, "3D Output Image with MSE, or if first parameter is "
                " 5D then this will be 4D with the MSE over time");
    vul_arg<string> a_mask("-m", "3 or 4D mask file (vol 0 will be used)");
    vul_arg<string> a_timeseries("-t", "timeseries prefix. Will print out each voxel's "
                "timseries. TRUE_<this> will be the timeseries from the true values, "
                "if first input is 5D then PARAM_[T]<this> will be the timeseries from "
                "the parameters at time T. If first input is 4D then this will just have "
                "one image for the single times worth of parameters");
    
    vul_arg<double> a_shortstep("-u", "micro-Step size", 1/128.);
    vul_arg<string> a_stimfile("-s", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<double> a_timestep("-m", "macro step size (how often to compare, "
                "can be anything down to the micro step size)", 2);
    vul_arg<double> a_simlength("-n", "Number of comparisons to make for"
                " simulation (so with macro step 3, and 300 comparisons,"
                " will run for 900 seconds worth of simulation ",300);
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("No Warning, just echoing");

    Label4DType::Pointer maskImage;
    Image4DType::Pointer pEst;
    Image4DType::Pointer pTrue;
    Image3DType::Pointer output;

    std::vector<Activation> input;

    /* Open up the input */
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_pEst() );
    reader->Update();
    pEst = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_pEst().c_str());
        exit(-1);
    }
    try {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_pTrue() );
    reader->Update();
    pTrue = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_pTrue().c_str());
        exit(-1);
    }
    
    if(!a_mask()) try {
        itk::ImageFileReader<Label4DType>::Pointer reader;
        reader = itk::ImageFileReader<Label4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_mask() );
        reader->Update();
        maskImage = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_mask().c_str());
        exit(-1);
    }

    unsigned int xlen = paramImage1->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = paramImage1->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = paramImage1->GetRequestedRegion().GetSize()[2];
    
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
                Image4DType::IndexType maskindex;
                Image4DType::IndexType img2index;;
                Image4DType::PointType point;
                paramImage1->TransformIndexToPhysicalPoint(index4, point);
                paramImage2->TransformPhysicalPointToIndex(point, img2index);
                maskImage->TransformPhysicalPointToIndex(point, maskindex);
                
                cout << "image1 index: " << index4 << endl;
                cout << "image2 index: " << img2index << endl;
                cout << "mask index: " << maskindex << endl;
                if(!paramImage2->GetRequestedRegion().IsInside(img2index) ||
                            !maskImage->GetRequestedRegion().IsInside(maskindex)
                            || maskImage->GetPixel(maskindex) == 0){
                    output->SetPixel(index3, 1);
                    continue;
                }

                vector<double> params1;
                vector<double> params2;
                fillvector(params1, paramImage1, index4);
                fillvector(params2, paramImage2, index4);
                list<double> sim1;
                list<double> sim2;
                simulate(sim1, params1, a_shortstep(), a_timestep(),input, 
                            a_simlength());
                simulate(sim2, params2, a_shortstep(), a_timestep(),input, 
                            a_simlength());
                output->SetPixel(index3, mse(sim1, sim2));
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



