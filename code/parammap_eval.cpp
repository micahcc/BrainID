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
            uint32_t length)
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

void analyzeHist(std::string filename, std::string outputdir,
            Image4DType::Pointer trueParams,
            Image4DType::Pointer preproc, 
            Label4DType::Pointer mask,
            std::vector<Activation>& stim,
            double TR, double shortTR)
{
    /* Create Output Images */
    cout << "Reading Histogram" << endl;
    itk::ImageFileReader<itk::OrientedImage<float, 6> >::Pointer reader;
    reader = itk::ImageFileReader<itk::OrientedImage<float, 6> >::New();
    reader->SetFileName( filename );
    reader->Update();
    itk::OrientedImage<float,6>::Pointer hist = reader->GetOutput();
    
    uint32_t xlen = hist->GetRequestedRegion().GetSize()[0];
    uint32_t ylen = hist->GetRequestedRegion().GetSize()[1];
    uint32_t zlen = hist->GetRequestedRegion().GetSize()[2];
    uint32_t tlen = hist->GetRequestedRegion().GetSize()[3];
    uint32_t Plen = hist->GetRequestedRegion().GetSize()[4];
    uint32_t Hlen = hist->GetRequestedRegion().GetSize()[5];

    for(uint32_t i = 0 ; i < 4 ; i++) {
        if(hist->GetRequestedRegion().GetSize()[i] != 
                    preproc->GetRequestedRegion().GetSize()[i])
            throw "SIZE MISMATCH!";
    }
    
    /* Calculated Paramater Map */
    Image4DType::SizeType size4 = {{xlen, ylen, zlen, 7}};
    Image4DType::Pointer trueParamR = Image4DType::New();
    trueParamR->SetRegions(size4);
    trueParamR->Allocate();
    trueParamR->FillBuffer(-1);
    
    /* Resampled Ground Pmap */
    Image4DType::Pointer gPmap;
    if(trueParams) {
        gPmap = Image4DType::New();
        gPmap->SetRegions(size4);
        gPmap->Allocate();
        gPmap->FillBuffer(-1);
    }
    
    /* MSE with actual data */
    Image3DType::SizeType size3 = {{xlen, ylen, zlen}};
    Image3DType::Pointer preprocMse = Image3DType::New();
    preprocMse->SetRegions(size3);
    preprocMse->Allocate();
    preprocMse->FillBuffer(-1);
    
    /* MSE with ground truth */
    Image3DType::Pointer trueMse = Image3DType::New();
    trueMse->SetRegions(size3);
    trueMse->Allocate();
    trueMse->FillBuffer(-1);
    
    /* Resampled Mask */
    Image3DType::Pointer maskR = Image3DType::New();
    maskR->SetRegions(size3);
    maskR->Allocate();
    maskR->FillBuffer(0);

    for(uint32_t xx = 0 ; xx < xlen ; xx++) {
        for(uint32_t yy = 0 ; yy < ylen ; yy++) {
            for(uint32_t zz = 0 ; zz < zlen ; zz++) {
                //initialize some variables
                itk::OrientedImage<float, 6>::IndexType index6 = 
                            {{xx, yy, zz, tlen-1,0,Hlen-1}};
                Image4DType::IndexType index4 = {{xx, yy, zz, 0}};
                Image4DType::PointType point;
                Image3DType::IndexType index3 = {{xx, yy, zz}};
                
                preproc->TransformIndexToPhysicalPoint(index4, point);
                
                /* Check Mask, and add point to re-sampled Mask */
                if(mask) {
                    Image4DType::IndexType maskindex;
                    mask->TransformPhysicalPointToIndex(point, maskindex);
                    if(!mask->GetRequestedRegion().IsInside(maskindex)
                            || mask->GetPixel(maskindex) == 0)
                        continue;
                    maskR->SetPixel(index3, mask->GetPixel(maskindex));
                }
                
                /* Output the parameter map that the particle filter put out */
                for(uint32_t ii = 0 ; ii < size4[3] ; ii++) {
                    index4[3] = ii;
                    index6[4] = ii;
                    trueParamR->SetPixel(index4, hist->GetPixel(index6));
                }
                 
                /* Calculate MSE with Particle Filter Input Image */
                vector<double> params1(size4[3]);
                for(uint32_t ii = 0 ; ii < size4[3] ; ii++) {
                    index6[4] = ii;
                    params1[ii] = hist->GetPixel(index6);
                }
                list<double> sim1;
                simulate(sim1, params1, shortTR, TR,stim, tlen);
                
                list<double> actual;
                for(uint32_t ii = 0 ; ii < tlen ; ii++) {
                    index4[3] = ii;
                    actual.push_front(preproc->GetPixel(index4));
                }
                
                preprocMse->SetPixel(index3, mse(sim1, actual));

                if(trueParams) {
                    Image4DType::IndexType realParamIndex;
                    trueParams->TransformPhysicalPointToIndex(point, realParamIndex);
                    if(!trueParams->GetRequestedRegion().IsInside(realParamIndex))
                        continue;
                    for(uint32_t ii = 0 ; ii < size4[3]; ii++) {
                        index4[3] = ii;
                        gPmap->SetPixel(index4, trueParams->GetPixel(realParamIndex));
                    }
                    vector<double> params2;
                    fillvector(params2, trueParams, index4);
                    list<double> sim2;
                    simulate(sim2, params2, shortTR, TR,stim, tlen);
                    trueMse->SetPixel(index3, mse(sim1, sim2));
                }
                
            }
        }
    }


}
            
/* Main Function */
int main(int argc, char* argv[])
{
    vul_arg<string> a_pEst(0, "File output of calc_parammap (see -D)");
    vul_arg<string> a_output(0, "output directory: <pmap.niigz> <mseReal.nii.gz> "
                "<maskresamp.nii.gz> <mseGround.nii.gz>");
    vul_arg<int> a_inputtype("-D", "0 - histogram, 1 - mean", 1);
    vul_arg<string> a_mask("-m", "3 or 4D mask file (vol 0 will be used)");

    vul_arg<string> a_pfilterInput("-p", "pfilter_input.nii.gz, the data that was fed into "
                "the particle filter");
    vul_arg<string> a_pTrue("-T", "True parameters, Either: \n\t\t4D param file, in the "
                "order: TAU_0, ALPHA, E_0, V_0, TAU_S, TAU_F, EPSILON, "
                "OR, \n\t\ttext file with parameters in the same order, one set");
    
    vul_arg<string> a_stimfile("-s", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<double> a_shortstep("-u", "micro-Step size", 1/1028.);
    vul_arg<double> a_timestep("-m", "macro step size (how often to compare, "
                "can be anything down to the micro step size)", 2.1);
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("No Warning, just echoing");

    Label4DType::Pointer maskImage;
    if(!a_mask().empty()) try {
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
    
    Image4DType::Pointer pTrue;
    if(!a_pTrue().empty()) try {
        itk::ImageFileReader<Image4DType>::Pointer reader;
        reader = itk::ImageFileReader<Image4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_pTrue() );
        reader->Update();
        pTrue= reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_pTrue().c_str());
        exit(-1);
    }
    
    Image4DType::Pointer iReal;
    if(!a_pfilterInput().empty()) try {
        itk::ImageFileReader<Image4DType>::Pointer reader;
        reader = itk::ImageFileReader<Image4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_pfilterInput() );
        reader->Update();
        iReal = reader->GetOutput();
        if(iReal->GetSpacing()[3] != a_timestep())
            fprintf(stderr, "Warning mismatched TR's\n");
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_pfilterInput().c_str());
        exit(-1);
    }
    
    std::vector<Activation> input;
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

    Image3DType::Pointer output;
    if(a_inputtype() == 0) {
        analyzeHist(a_pEst(), a_output(), pTrue, iReal, maskImage, input,
                    a_timestep(), a_shortstep());
    } else if(a_inputtype() == 1) {
        //analyzeParamMean(a_pEst(), maskImage, input)
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



