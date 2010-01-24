#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"
#include "tools.h"

#include <itkMultiplyImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkRandomImageSource.h>
#include <itkExtractImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include "modNiftiImageIO.h"

#include <sstream>
#include <iostream>
#include <vector>

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

Label4DType::Pointer createRegions(double sigma, double threshp,
            Image4DType::Pointer templ = NULL)
{
    itk::RandomImageSource<Image4DType>::Pointer rImage = 
                itk::RandomImageSource<Image4DType>::New();
    rImage->SetNumberOfThreads(10);
    rImage->SetMax(1);
    rImage->SetMin(0);
    itk::BinaryThresholdImageFilter<Image4DType, Image4DType>::Pointer thresh = 
                itk::BinaryThresholdImageFilter<Image4DType, Image4DType>::New();

    unsigned long out_size[4];
    if(templ) {
        out_size[0] = templ->GetRequestedRegion().GetSize()[0];
        out_size[1] = templ->GetRequestedRegion().GetSize()[1];
        out_size[2] = templ->GetRequestedRegion().GetSize()[2];
        out_size[3] = 1;
    } else {
        out_size[0] = 100;
        out_size[1] = 100;
        out_size[2] = 100;
        out_size[3] = 1; 
        
    }
    //outimage
    rImage->SetSize(out_size);
    
    /* Create a Random Field, Perform Smoothing to the set number of resels
     * then threshold, to create the regions
     */
    itk::DiscreteGaussianImageFilter<Image4DType, Image4DType>::Pointer gaussF = 
                itk::DiscreteGaussianImageFilter<Image4DType, Image4DType>::New();
    double variance[3] = {sigma,sigma,sigma};
    gaussF->SetVariance(variance);
    gaussF->SetInput(rImage->GetOutput());
    
    thresh->SetInput(gaussF->GetOutput());
    thresh->SetLowerThreshold(0);
    thresh->SetUpperThreshold(threshp);
    thresh->SetInsideValue(0);
    thresh->SetOutsideValue(1);
    

    //segment image into discrete blobs
    itk::ConnectedComponentImageFilter<Image4DType, Label4DType>::Pointer conn = 
                itk::ConnectedComponentImageFilter<Image4DType, Label4DType>::New();
    conn->SetInput(thresh->GetOutput());

    conn->Update();
    
    if(templ) {
        conn->GetOutput()->SetDirection(templ->GetDirection());
        conn->GetOutput()->SetSpacing(templ->GetSpacing());
    }
    return conn->GetOutput();
}

vector< vector<double> > read_params(string filename) 
{
    FILE* fin = fopen(filename.c_str(), "r");
    
    vector< vector<double> > params;
    if(!fin) 
        return params;

    char* input = NULL;
    size_t size = 0;
    char* prev = NULL;
    char* curr = NULL;
    double parsed = 0;
    while(getline(&input, &size, fin) && !feof(fin)) {
        vector<double> tmp;
        parsed = strtod(input, &curr);
        while(prev != curr) {
            printf("%f, ", parsed);
            tmp.push_back(parsed);
            prev = curr;
            parsed = strtod(prev, &curr);
        }

        params.push_back(tmp);
        free(input);
        input = NULL;
        printf("%zu\n", params[params.size()-1].size());
    }
    fclose(fin);
    return params;
}

/*Modifies state, by steping forward delta in time*/
int transition(State& state, const vector<double>& params, double delta, double in)
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

/*Performs readout from state, parameters*/
double readout(State& state, const vector<double>&params)
{
//    fprintf(stdout, "%f %f %f\n", params.at(V_0), params.at(A_1), params.at(A_2));
    return params.at(V_0)*(params.at(A_1)*(1-state.Q)-params.at(A_2)*(1-state.V));
}

/*Simulates an entire timeseries for a set of parameters*/
vector<double> simulate(const vector<Activation>& activations, 
            const vector<double>& params, double endtime,
            double timestep, double int_timestep)
{
    State state;
    state.V = 1;
    state.Q = 1;
    state.F = 1;
    state.S = 0;

    double value;
    int shorttime = 0;
    int longtime = 0;
    size_t act_pos = 0;
    double act_level = 0;

    vector<double> out(endtime/timestep);
    while(longtime < (int)out.size()) {
        /*Update Input If there is a change for the current time*/
        act_level = 0; //this results in transients, rather than squares
        while(act_pos < activations.size() && activations[act_pos].time <= 
                        shorttime*int_timestep) {
            act_level = activations[act_pos].level;
            printf("Act: %f %f\n", shorttime*int_timestep , act_level);
            act_pos++;
        }

        /*Update state variables*/
        int res = transition(state, params, int_timestep, act_level);
        switch(res) {
            case -1:
                printf("Error in s\n");
                break;
            case -2:
                printf("Error in f\n");
                break;
            case -3:
                printf("Error in v\n");
                break;
            case -4:
                printf("Error in q\n");
                break;
        }

        /*If it is time to sample, do so*/
        if(shorttime*int_timestep >= longtime*timestep) {
            value = readout(state, params);
            out[longtime] = value;
            longtime++;
        }
        shorttime++;
    }
    return out;
}

Image4DType::Pointer simulate(string param_f, string act_f, 
            Label4DType::Pointer regions, double endtime, double timestep, 
            double int_timestep)
{
    Image4DType::Pointer out = Image4DType::New();
    Image4DType::SizeType size = regions->GetRequestedRegion().GetSize();
    
    /*Determine Size*/
    size[3] = endtime/timestep;
    out->SetRegions(size);
    out->Allocate();
    out->SetSpacing(regions->GetSpacing());
    out->SetDirection(regions->GetDirection());

    //open and read param_f
    vector< vector<double> > params = read_params(param_f);
    
    //open act_f and read act_f
    vector<Activation> activations = read_activations(act_f.c_str());

    vector< vector<double> > timeseries(params.size());
    for(int i = 0 ; i < (int)timeseries.size() ; i++) {
        timeseries[i] = simulate(activations, params[i], endtime, timestep, 
                    int_timestep);
        printf("timeseries %i\n", i);
        for(int j = 0 ; j < (int)timeseries[i].size(); j++) {
            printf("%e\n", timeseries[i][j]);
        }
        printf("\n");
    }


    //simulate each region, using act_f for activation times
    for(size_t xx = 0 ; xx<regions->GetRequestedRegion().GetSize()[0] ; xx++) {
        for(size_t yy = 0 ; yy<regions->GetRequestedRegion().GetSize()[1] ; yy++) {
            for(size_t zz = 0 ; zz<regions->GetRequestedRegion().GetSize()[2] ; zz++) {
                Image4DType::IndexType index = {{xx, yy, zz, 0}};
                int i = (int)regions->GetPixel(index)-1;
                if(i >= 0) {
                    /*Write vector to image*/
                    writeVector<double, std::vector<double> >(out, 3, 
                                timeseries[i%params.size()], index);
                }
            }
        }
    }

    return out;
}

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_templ("-t" ,"Template (takes orientation/spacing/brain)");
    vul_arg<string> a_active("-a" ,"Activation file: <onset> <scale>");
    vul_arg<string> a_imageOut("-oi" ,"Image Out", "timeseries.nii.gz");
    vul_arg<string> a_timesOut("-ot" ,"Image Times Out", "timeseries.txt");
    vul_arg<double> a_thresh("-T" ,"Threshold for region gen.", .53);
    vul_arg<double> a_sigma("-s" ,"Sigma for gaussian filter, region gen", 4);
    vul_arg<string> a_params("-p" ,"File with parameters, 1 region per line, "
                "space delim: TAU_0 ALPHA E_0 V_0 TAU_S TAU_F EPSILON A_1 A_2");
    vul_arg<double> a_endtime("-e", "End Time");
    vul_arg<double> a_timestep("-l", "Large, Output Time Steps", 3);
    vul_arg<double> a_utimestep("-m", "Micro, Simulation Time Steps", .001);
    
    /* Processing */
    vul_arg_parse(argc, argv);
    
    Image4DType::Pointer templ = NULL;
    
    /* Template Image */
    if(a_templ() != "") {
        itk::ImageFileReader<Image4DType>::Pointer reader = 
                    itk::ImageFileReader<Image4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader ->SetFileName( a_templ() );
        templ = reader->GetOutput();
        templ->Update();
    }
    
    Label4DType::Pointer regions = createRegions(a_sigma(), a_thresh(), templ);
    printf("\nSpacing %f %f %f\n", regions->GetSpacing()[0],regions->GetSpacing()[1], 
                regions->GetSpacing()[2]);
    
    {
    fprintf(stderr, "Outputing regions\n");
    itk::ImageFileWriter<Label4DType>::Pointer writer = 
                itk::ImageFileWriter<Label4DType>::New();
    writer->SetInput(regions);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName("regions.nii.gz");
    writer->Update();
    }

    fprintf(stderr, "Creating/Filling Timeseries\n");
    Image4DType::Pointer out = simulate(a_params(), a_active(), regions,
                a_endtime(), a_timestep(), a_utimestep());
    
    printf("\nSpacing %f %f %f\n", out->GetSpacing()[0],out->GetSpacing()[1], 
                out->GetSpacing()[2]);

    fprintf(stderr, "Writing...\n");
    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetInput(out);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(a_imageOut());
    writer->Update();
    fprintf(stderr, "Done\n");

    fprintf(stderr, "Writing output time file\n");
    FILE* fout = fopen(a_timesOut().c_str(), "w");
    if(ferror(fout)) {
        fprintf(stderr, "Failed to open file for writing\n");
        return -3;
    }

    for(int i = 0 ; i*a_timestep() < a_endtime(); i++) {
        fprintf(fout, "%f %f\n", i*a_timestep(), 1.0);
    }
        
    return 0;
}
