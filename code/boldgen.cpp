#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"
#include "BoldModel.hpp"

#include <indii/ml/aux/vector.hpp>

#include <vul/vul_arg.h>
#include <vcl_list.h>

#include "modNiftiImageIO.h"
#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <sstream>
#include <cmath>

using namespace std;

const int SERIES_DIR = 0;
const int PARAM_DIR = 1;
const int TIME_DIR = 3;

typedef itk::OrientedImage<double, 4> Image4DType;

void init4DImage(Image4DType::Pointer& out, size_t xlen, size_t ylen, 
            size_t zlen, size_t tlen)
{
    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index;
    Image4DType::SizeType out_size;

    out_size[0] = xlen;
    out_size[1] = ylen;
    out_size[2] = zlen;
    out_size[3] = tlen; 
    
    out_index[0] = 0;
    out_index[1] = 0;
    out_index[2] = 0;
    out_index[3] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    out->SetRegions( out_region );
    out->Allocate();
}

//dir1 should be the direction of several separate series
//dir2 should be the direction that you want to get rms of
//RMS for a non-zero mean signal is 
//sqrt(mu^2+sigma^2)
void get_rms(Image4DType::Pointer in, size_t dir1, size_t dir2, 
            vector<double>& out, int series)
{
    out.assign(in->GetRequestedRegion().GetSize()[dir1], 0);
    
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                iter(in, in->GetRequestedRegion());
    iter.SetFirstDirection(dir1);
    iter.SetSecondDirection(dir2);
    iter.GoToBegin();

    int numelements = in->GetRequestedRegion().GetSize()[dir2];

    vector<double> mean(out.size(), 0);
    /* get average */
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        while(!iter.IsAtEndOfLine()) {
            mean[ii] += iter.Get();
            ii++;
            ++iter;
        }
        iter.NextLine();
    }
    for(size_t ii= 0 ; ii<mean.size() ; ii++) {
        mean[ii] /= numelements;
    }

    /* variance */
    iter.GoToBegin();
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        while(!iter.IsAtEndOfLine()) {
            out[ii] += pow(iter.Get()-mean[ii],2);
            ii++;
            ++iter;
        }
        iter.NextLine();
    }
    for(size_t ii= 0 ; ii<out.size() ; ii++)
        out[ii] /= numelements;


    //sqrt(mu^2 + var)
    for(size_t ii = 0 ; ii < out.size() ; ii++) {
        out[ii] = sqrt(pow(mean[ii], 2) + out[ii]);
    }
}

void add_noise(Image4DType::Pointer in, double snr, gsl_rng* rng, int series) 
{
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                iter(in, in->GetRequestedRegion());
    iter.SetFirstDirection(SERIES_DIR);
    iter.SetSecondDirection(TIME_DIR);
    iter.GoToBegin();
    
    vector<double> rms(series, 0);

    //calculate rms of image over time
    get_rms(in, SERIES_DIR, TIME_DIR, rms, series);
    for(size_t ii= 0 ; ii<rms.size() ; ii++) {
        cout << "RMS: " << ii << ":" << rms[ii] << endl;
    }

    iter.GoToBegin();
    while(!iter.IsAtEndOfSlice()) {
        size_t ii=0;
        //move through series'
        while(!iter.IsAtEndOfLine()) {
            iter.Value() += gsl_ran_gaussian(rng, rms[ii]/sqrt(snr));
            //change series
            ++iter;
            ii++; 
        }
        //move through time
        iter.NextLine();
    }
    
    get_rms(in, SERIES_DIR, TIME_DIR, rms, series);
    for(size_t ii= 0 ; ii<rms.size() ; ii++) {
        cout << "RMS: " << ii << " " << rms[ii] << endl;
    }
}

int main (int argc, char** argv)
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);
    srand(1333);

    vul_arg<string> a_boldfile("-bf", "boldfile to write to (image) - boldfine with noise"
                " will be the same name but with \"noise-\" prefixed", "");
    vul_arg<double> a_outstep("-ot", "How often to sample", 2);
    vul_arg<double> a_simstep("-st", "Step size for sim, smaller is more accurate",
                .001);
    vul_arg<double> a_stoptime("-end", "What time to end at", 600);
    vul_arg<unsigned> a_series("-n", "Number of brain regions to simulate", 1);
    vul_arg<string> a_statefile("-sf", "file to write out state data to", "");
    vul_arg<string> a_stimfile("-istim", "file to read in stimuli from", "");
    vul_arg<bool> a_randstim("-randstim", "create a random stimulus", false);
    vul_arg<string> a_randstim_file("-rstimfile", "where to write to, for "
                " stim generation", "");
    vul_arg<double> a_randstim_t("-rstimt", "time between changes, for stim "
                "generation", 4);
    vul_arg<double> a_randstim_p("-rstimp", "probability of high stimulus, for "
                "stim generation", .5);
    vul_arg<double> a_noise_snr("-snr", "SNR of Gaussian Noise to apply to bold"
                " signal", 0);
    vul_arg<string> a_paramfile("-X0file", "File with initial conditions for "
                "simulation", "");
    vul_arg< vcl_vector<double> > a_params("-X0", "Inital conditions for "
                "simulation");

    vul_arg_parse(argc,argv);

    BoldModel model(false, a_series());
    
    //create a 4D output image of appropriate size.
    itk::ImageFileWriter< Image4DType >::Pointer writer = 
        itk::ImageFileWriter< Image4DType >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    Image4DType::Pointer measImage = Image4DType::New();
    
    //TODO deal with add error in double which could cause less or more
    //states to be simulated
    //tlen = |T|T|T| + one for the series number
    init4DImage(measImage, a_series(), 1, 1, (int)ceil(a_stoptime()/a_outstep()));
  
    itk::MetaDataDictionary dict = measImage->GetMetaDataDictionary();
    itk::EncapsulateMetaData<double>(dict, "TemporalResolution", a_outstep());
    itk::EncapsulateMetaData<unsigned int>(dict, "NumSections", a_series());
    itk::EncapsulateMetaData<string>(dict, "Dim3", "time");
    itk::EncapsulateMetaData<string>(dict, "Dim0", "slices");
    //fill in mapping of Section Index to number section number i+5
    for(size_t i=0 ; i<a_series() ; i++ ) {
        ostringstream oss;
        oss.str("");
        oss << "MapIndex " << i;
        itk::EncapsulateMetaData<unsigned int>(dict, oss.str(), i+5);
    }
    measImage->SetMetaDataDictionary(dict);
    Image4DType::SpacingType space4;
    for(int i = 0 ; i < 4 ; i++) space4[i] = 1;
    space4[TIME_DIR] = a_outstep();
    measImage->SetSpacing(space4);

    //initialize first line in t direction to hold the section number
    
    Image4DType::Pointer outState = Image4DType::New();
    init4DImage(outState, a_series(), model.getStateSize(), 1, 
                (int)ceil(a_stoptime()/a_outstep()));

    itk::EncapsulateMetaData<string>(dict, "Dim1", "state");
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexTauS",    0);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexTauF",    1);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexEpsilon", 2);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexTau0",    3);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexAlpha",   4);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexE0",      5);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexV0",      6);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexVT",      7);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexQT",      8);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexST",      9);
    itk::EncapsulateMetaData<unsigned int>(dict, "IndexFT",     10);
 
    outState->SetMetaDataDictionary(dict);

//#ifdef TEST
//    unsigned int tmp1000;
//    dict = outState->GetMetaDataDictionary();
//    itk::ExposeMetaData<unsigned int>(dict, "IndexE0", tmp1000);
//    cout << "IndexE0 " << tmp1000 << endl;;
//    dict.Print(cout);
//#endif

    //setup iterators
    itk::ImageSliceIteratorWithIndex<Image4DType> 
                meas_it(measImage, measImage->GetRequestedRegion());
    meas_it.SetFirstDirection(SERIES_DIR);
    meas_it.SetSecondDirection(TIME_DIR);
    meas_it.GoToBegin();
    
    itk::ImageLinearIteratorWithIndex<Image4DType> 
                state_it(outState, outState->GetRequestedRegion());
    state_it.SetDirection(PARAM_DIR);

    //Used to add noise
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (int)((time(NULL)*rank)/11.));

//    outputVector(std::cout, systemstate);
//    std::cout << std::endl;

    aux::vector systemstate(model.getStateSize());
    if(!a_params().empty()) {
        if(systemstate.size() != a_params().size()) {
            cerr << "Error invalid number of parameters for -X0, should give " 
                        << systemstate.size()<< endl;
            exit(-2);
        }

        for(unsigned int i = 0 ; i<systemstate.size() ; i++)
            systemstate[i] = a_params()[i];
    
    } else if(!a_paramfile().empty()) {
        cout << "Reading Simulation Init/theta from: " 
                    << a_paramfile() << endl;
        
        ifstream init(a_paramfile().c_str());
        
        for(unsigned int i = 0 ; i < model.getStateSize() ; i++)
            init >> systemstate[i];

        if(init.eof()) {
            cerr << "Error not enough arguments given in file" << endl;
            outputVector(std::cout, systemstate);
            cerr << endl;
            exit(-1);
        }
    } else {
        cout << "Using random values for init/theta" << endl;
        aux::DiracMixturePdf x0(model.getStateSize());
        model.generatePrior(x0, 10000);
        systemstate = x0.sample();

#ifdef ZEROSTART
        //this assumes you start at resting state, which is not a bad
        //assumption. Plus this way you don't get negative bold.
        for(int i = BoldModel::THETA_SIZE ; i < model.getStateSize() ; i++) {
            if((i-BoldModel::THETA_SIZE)%BoldModel::STATE_SIZE == 2) 
                systemstate[i] = 1;
            else
                systemstate[i] = 0;
        }
#endif //ZEROSTART
    } 

    outputVector(cout, systemstate);
    cout << endl;

    int sample = 0;
    int count = 0;
    double realt = 0;
    double prev = 0;

    ofstream fout;
    ifstream fin;

    //TODO implement multiple series (based on noise)
    aux::vector input(1);
    input[0] = 0;
    double nextinput;
    int endcount = a_stoptime()/a_simstep();

    //first try to open the input file
    if(!a_stimfile().empty()) {
        fin.open(a_stimfile().c_str());
        if(!fin.is_open()) {
            fprintf(stderr, "Error bad input file for stimulus\n");
            exit(-1);
        }
        fin >> nextinput;
    //if there is no input then open the output randstim
    } else if(!a_randstim_file().empty())  {
        fout.open(a_randstim_file().c_str());
        if(!fout.is_open()) {
            fprintf(stderr, "Error bad output file for stimulus: %s\n",
                        a_randstim_file().c_str());
            exit(-2);
        }
        input[0] = (double)rand()/RAND_MAX < a_randstim_p();
        fout << 0 << " " << input[0] << endl;
        nextinput = a_randstim_t();
    }

//    cout << endcount << endl;
    for(count = 0 ; count  < endcount; count++) {
        //setup next timestep
        prev = realt;
        realt = count*a_simstep();
        if(fin.is_open() && !fin.eof() && realt >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
        } else if(fout.is_open() && realt >= nextinput) {
            input[0] = (double)rand()/RAND_MAX < a_randstim_p();
            fout << nextinput << " " << input[0] << endl;
            nextinput += a_randstim_t();
        }

        int returnv;
        if((returnv = model.transition(systemstate, realt, realt-prev, input)) != 0) {
            cout << returnv;
            exit(returnv);
        }
        //TODO add noise to simulation
        
        //check to see if it is time to sample
        if(realt > sample*a_outstep()) {
//            fprintf(stderr, "Sample: %i, Time: %f\n", sample, sample*a_outstep());
            Image4DType::IndexType index = {{ 0, 0, 0, 0 }};
            index[TIME_DIR] = sample;
            writeVector<double>(measImage, SERIES_DIR, model.measure(systemstate),
                        index);
            writeVector<double>(outState, PARAM_DIR, systemstate, index);
            
            sample++;;
        }
        
    }
    
    if(!a_boldfile().empty()) {
        writer->SetFileName(a_boldfile());  
        writer->SetInput(measImage);
        writer->Update();
    }

    if(fout.is_open()) {
        fout.close();
    }
    
    if(a_noise_snr() != 0) {
        add_noise(measImage, a_noise_snr(), rng, a_series());
        if(!a_boldfile().empty()) {
            ostringstream oss("");
            oss << "noise-" << a_boldfile();
            writer->SetFileName(oss.str());  
            writer->SetInput(measImage);
            writer->Update();
        }
    }
    
    if(!a_statefile().empty()) {
        writer->SetFileName(a_statefile());  
        writer->SetInput(outState);
        writer->Update();
    }
  
    gsl_rng_free(rng);
    return 0;
}

