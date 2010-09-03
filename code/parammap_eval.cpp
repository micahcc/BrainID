#include <boost/math/special_functions/fpclassify.hpp>

#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkMetaDataObject.h>

#include "tools.h"
#include "modNiftiImageIO.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <string>

#include <vul/vul_arg.h>

using namespace std;

int SUCCESS = 2;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;

namespace aux = indii::ml::aux;

/* Helper function to write an image "out" to prefix + filename */
template <typename T>
void writeImage(std::string base, std::string name, typename T::Pointer in)
{
    base.append(name);
    std::cout << "Writing " << base << std::endl;
    typename itk::ImageFileWriter<T>::Pointer writer;
    writer = itk::ImageFileWriter<T>::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName( base );
    writer->SetInput( in );
    std::cout << "Done " << std::endl;
    writer->Update();
}

/* Read an image from base+name and returns the T dimensional image */
template <typename T>
typename T::Pointer readImage(std::string base, std::string name)
{
    base.append(name);
    std::cout << "Reading " << base << std::endl;
    typename itk::ImageFileReader<T>::Pointer reader;
    reader = itk::ImageFileReader<T>::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( base );
    reader->Update();
    std::cout << "Done " << std::endl;
    return reader->GetOutput();
}

/* Reads the output given a state and set of parameters, part of the Bold Model */
double readout(State& state, const std::vector<double>& params)
{
    return params[V_0]*(params[A_1]*(1-state.Q)-params[A_2]*(1-state.V));
}

//typedef int v4sf __attribute__ ((mode(V4SF))); // vector of four single floats

#include <xmmintrin.h>
union d2vector 
{
  __v2df v;
  double f[2];
};

/* Integrates state over time of delta */
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
    return 0;
}

void printError(int err, std::vector<double>& params, State& state, 
            double rt, double prev_rt)
{
    switch(err) {
        case -1:
            printf("Error in s at %f - %f\n", rt, prev_rt);
            printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
            printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                    "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                    params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                    params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
            throw(err);
        case -2:
            printf("Error in f at %f - %f\n", rt, prev_rt);
            printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
            printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                    "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                    params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                    params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
            throw(err);
        case -3:
            printf("Error in v at %f - %f\n", rt, prev_rt);
            printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
            printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                    "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                    params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                    params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
            throw(err);
        case -4:
            printf("Error in q at %f - %f\n", rt, prev_rt);
            printf("State: V:%f Q:%f F:%f S:%f\n", state.V, state.Q, state.F, state.S);
            printf("Params: TAU_0=%f, ALPHA=%f, E_0=%f, V_0=%f, TAU_S=%f, "
                    "TAU_F=%f, EPSILON=%f, A_1=%f, A_2=%f\n", params[TAU_0], 
                    params[ALPHA], params[E_0], params[V_0], params[TAU_S], 
                    params[TAU_F], params[EPSILON], params[A_1], params[A_2]);
            throw(err);
        default:
        break;
    }

}

/* Simulates an FMRI image given a set of parameters from paramImg. Simulations
 * only performed in locations where mask > 0. 
 * input_vector - is the stimulus sequence
 * dt_s         - short steps used in integration
 * dt_l         - long steps used to decide how often to measure
 * count        - the number of measurements in the output image (decides run time)
*/
Image4DType::Pointer simulate(Image4DType::Pointer paramImg, Label3DType::Pointer mask,
            const std::vector<Activation>& input_vector, double dt_s, double dt_l,
            uint32_t count)
{
    itk::ImageLinearIteratorWithIndex<Label3DType> itL(mask, mask->GetRequestedRegion());
    Label3DType::IndexType index3 = {{0,0,0}};
    Image4DType::IndexType index4;

    std::vector<double> params(paramImg->GetRequestedRegion().GetSize()[3]);
    
    Image4DType::Pointer out = Image4DType::New();
    Image4DType::SizeType size4 = paramImg->GetRequestedRegion().GetSize();
    size4[3] = count;
    out->SetRegions(size4);
    out->Allocate();
    out->FillBuffer(0);
    
    //simulate all that match
    while(!itL.IsAtEnd()) {
        while(!itL.IsAtEndOfLine()) {
            if(itL.Value() != SUCCESS) {
                ++itL;
                continue;
            }

            index3 = itL.GetIndex();
            for(uint32_t ii = 0 ; ii < 3 ; ii++)
                index4[ii] = index3[ii];
            for(index4[3] = 0 ; index4[3] < (int)params.size() ; index4[3]++) {
                params[index4[3]] = paramImg->GetPixel(index4);
            }
            
            std::cout << index3 << " ";

            //Set up simulation state, initialize
            State state;
            state.V = 1;
            state.Q = 1;
            state.F = 1;
            state.S = 0;
            index4[3] = 0;

            //Counters/temps
            double input = 0;
            size_t input_index = 0;
            double rt = 0;
            double prev_rt = 0;
            try {
            for(size_t ii = 0 ; index4[3] != count; ii++) {
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
//                printError(res, params, state, rt, prev_rt);

                /*If it is time to sample, do so*/
                while(dt_l*index4[3] <= rt) {
                    if(boost::math::isnan<double>(readout(state, params)) || 
                                boost::math::isinf<double>(readout(state,params))) {
                        throw(-2);
                    }
                    out->SetPixel(index4, readout(state, params));
                    index4[3]++;
                }
            }
            } catch(...) {
                std::cout << "Skipping " << index4 << std::endl;
            }
            ++itL;
        }
        itL.NextLine();
    }
    return out;
}

/* Main Function */
int main(int argc, char* argv[])
{
    vul_arg<string> a_dir(0, "Directory to evaluate, needs: stim pfilter_input.nii.gz,"
                " statuslabel.nii.gz, parammu_f.nii.gz [parammap.nii.gz]\nWill be written"
                " to the directory:\nest_input.nii.gz mse.nii.gz act.nii.gz [mse_true.nii.gz]");

    vul_arg<double> a_shortstep("-u", "micro-Step size", 1/2048.);
    vul_arg<double> a_timestep("-m", "macro step size (how often to compare, "
                "can be anything down to the micro step size)", 2.1);
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("Parameters");

    Label3DType::Pointer statusImg = readImage<Label3DType>(a_dir(), 
                "statuslabel.nii.gz");
    Image4DType::Pointer estParamImg = readImage<Image4DType>(a_dir(), 
                "parammu_f.nii.gz");
    Image4DType::Pointer realTSImg = readImage<Image4DType>(a_dir(), 
                "pfilter_input.nii.gz");
    std::cout << "Don't forget to shift stim properly according to pfilter_input" 
                << std::endl;
    
    if(abs(realTSImg->GetSpacing()[3] - a_timestep()) > .0001)
        std::cout << "Warning mismatch between pfilter_input and -m option"
                    << std::endl;

    uint32_t tlen = realTSImg->GetRequestedRegion().GetSize()[3];

    Image4DType::Pointer trueParamImg;
    try{
        trueParamImg = readImage<Image4DType>(a_dir(), "parammap.nii.gz");
        std::cout << "Found true parameter map: " << a_dir() << "parammap.nii.gz" << std::endl;
        //todo, need to re-sample trueParamImg
    } catch (...) {
        std::cout << "...none there. If you want to compare with real " 
                    << "parameters you need to add parammap.nii.gz to the dir" << std::endl;
    }
    
    /* Open Stimulus file */
    std::vector<Activation> input= read_activations(a_dir(), "stim");
    if(input.empty()) 
        return -1;

    if(input.size() == 0) {
        Activation tmp;
        tmp.time = 0;
        tmp.level= 0;
        input = std::vector<Activation>(1,tmp);
    }
    
    /* 
     * Calculate the MSE between the simulation with estimated parameters and
     * the preprocessed real data
     */
    Image4DType::Pointer estSim = simulate(estParamImg, statusImg, input, 
                a_shortstep(), a_timestep(), tlen);
    copyInformation<Image4DType, Image4DType>(realTSImg, estSim);
    writeImage<Image4DType>(a_dir(), "est_input.nii.gz", estSim);
    
    Image3DType::Pointer estMse = mse(estSim, realTSImg);
    copyInformation<Image4DType, Image3DType>(realTSImg, estMse);
    writeImage<Image3DType>(a_dir(), "mse.nii.gz", estMse);
        
    Image3DType::Pointer estMI = mutual_info(6,6, realTSImg, estSim);
    copyInformation<Image4DType, Image3DType>(realTSImg, estMI);
    writeImage<Image3DType>(a_dir(), "mi.nii.gz", estMI);

    Image3DType::Pointer mad = median_absolute_deviation(realTSImg);
    copyInformation<Image4DType, Image3DType>(realTSImg, mad);
    writeImage<Image3DType>(a_dir(), "mad.nii.gz", mad);
    
    /* 
     * Calculate the MSE between the simulation with estimated parameters and
     * the simulation with the true paramters
     */
    if(trueParamImg) {
        typedef itk::NearestNeighborInterpolateImageFunction<Image4DType, double> InterpT;
        typedef itk::ResampleImageFilter<Image4DType, Image4DType, double> ResampT;
        InterpT::Pointer interp = InterpT::New();
        ResampT::Pointer resampler = ResampT::New();
        resampler->SetInterpolator(interp);
        resampler->SetInput(trueParamImg);
        resampler->SetOutputParametersFromImage(estParamImg);
        resampler->Update();
        writeImage<Image4DType>(a_dir(), "parammap_resamp.nii.gz", resampler->GetOutput());
        Image4DType::Pointer trueParamImg2 = resampler->GetOutput();
        
        Image4DType::Pointer trueSim = simulate(trueParamImg2, statusImg, input,
                    a_shortstep(), a_timestep(), tlen);
        copyInformation<Image4DType, Image4DType>(realTSImg, trueSim);
        writeImage<Image4DType>(a_dir(), "true_bold.nii.gz", trueSim);
        
        Image3DType::Pointer trueMse = mse(trueSim, estSim);
        copyInformation<Image4DType, Image3DType>(realTSImg, trueMse);
        writeImage<Image3DType>(a_dir(), "mse_true.nii.gz", trueMse);

        Image3DType::Pointer trueMI = mutual_info(6,6, trueSim, estSim);
        copyInformation<Image4DType, Image3DType>(realTSImg, trueMI);
        writeImage<Image3DType>(a_dir(), "mi_true.nii.gz", trueMI);
        
    }

    return 0;

}



