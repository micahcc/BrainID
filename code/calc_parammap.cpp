//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan
#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkMetaDataObject.h>
#include <itkDivideImageFilter.h>
#include <itkSubtractImageFilter.h>

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "tools.h"
#include "modNiftiImageIO.h"
#include "BoldPF.h"
#include "callbacks.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;
typedef itk::DivideImageFilter< Image4DType, Image4DType, Image4DType > DivF4;
typedef itk::SubtractImageFilter< Image4DType > SubF4;

namespace aux = indii::ml::aux;
typedef indii::ml::filter::ParticleFilter<double> Filter;

bool checkmask(Label4DType::Pointer maskimg, Image4DType::PointType point)
{
    if(!maskimg) return true;

    Image4DType::IndexType index;
    maskimg->TransformPhysicalPointToIndex(point, index);
    if(maskimg->GetRequestedRegion().IsInside(index) && 
                    maskimg->GetPixel(index) > 0) {
        return true;
    }

    return false;
}

void fillvector(std::vector< aux::vector >& output, Image4DType* input,
            Image4DType::IndexType pos, bool delta)
{       
    ImgIter iter(input, input->GetRequestedRegion());
    iter.SetDirection(3);
    iter.SetIndex(pos);

    output.resize(input->GetRequestedRegion().GetSize()[3]);
    if(delta) {
        int i = 1;
        double prev = iter.Get();
        ++iter;
        output[0] = aux::vector(1, 0);
        while(!iter.IsAtEndOfLine()) {
            output[i] = aux::vector(1, iter.Get()-prev);
            prev = iter.Get();
            ++iter;
            i++;
        }
    } else {
        int i = 0;
        while(!iter.IsAtEndOfLine()) {
            output[i] = aux::vector(1, iter.Get());
            ++iter;
            i++;
        }
    }
}

Image4DType::Pointer preprocess_help(Image4DType::Pointer input, 
            std::vector<Activation>& stim, double sampletime, unsigned int erase,
            bool nospline, bool smart)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    /* Set up measurements image */
    //*output << "Conditioning FMRI Image" << endl;
    //remove first 2 time step, since they are typically polluted
    input = pruneFMRI(input, stim, sampletime, erase);

    //calculate %difference, which is used normally for the bold signal
    // or the modified % difference (with spline rather than mean)
    if(nospline) {
        std::cerr << "Changing Image to %difference" << std::endl;
        SubF4::Pointer sub = SubF4::New();   
        DivF4::Pointer div = DivF4::New();
        Image4DType::Pointer mean = extrude(Tmean(input),
                    input->GetRequestedRegion().GetSize()[3]);
        sub->SetInput1(input);
        sub->SetInput2(mean);
        div->SetInput1(sub->GetOutput());
        div->SetInput2(mean);
        div->Update();
        input = div->GetOutput();
    } else if(smart){
        std::cerr << "De-trending, then dividing by mean" << endl;
        input = deSplineByStim(input, 10, stim, sampletime);
    } else {
        std::cerr << "De-trending, then dividing by mean" << endl;
        input = deSplineBlind(input, input->GetRequestedRegion().GetSize()[3]/80);
    }
    std::cerr << "Done." << endl;

    /* Save detrended image */
    if(rank == 0) try {
        itk::ImageFileWriter<Image4DType>::Pointer out = 
                    itk::ImageFileWriter<Image4DType>::New();
        out->SetInput(input);
        out->SetFileName("pfilter_input.nii.gz");
        cout << "Writing: " << "pfilter_output.nii.gz" << endl;
        out->Update();
    } catch(itk::ExceptionObject) {
        cerr << "Error opening pfilter_input.nii.gz" << endl;
        exit(-4);
    }
    return input;
}

/* Main Function */
int main(int argc, char* argv[])
{
    /* Initialize mpi */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    vul_arg<string> a_input(0, "4D timeseries file");
    
    vul_arg<unsigned> a_level("-a", "amount of output: 0 - basics, 1 - all particles", 0);
    vul_arg<string> a_mask("-m", "3D mask file");
    vul_arg<bool> a_dc("-c", "Calculate DC gain as a state variable", false);
    vul_arg<bool> a_delta("-l", "Use deltas between measurements, this precludes"
                "the drift option", false);
    vul_arg<bool> a_smart("-S", "Use \"smart\" knots based on less active regions"
                , false);
    
    vul_arg<unsigned> a_num_particles("-p", "Number of particles.", 3000);
    vul_arg<unsigned> a_divider("-d", "Intermediate Steps between samples.", 128);
    vul_arg<string> a_stimfile("-s", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<bool> a_expweight("-e", "Use exponential weighting function", false);
    vul_arg<double> a_timestep("-t", "TR (timesteps in 4th dimension)", 2);
    
    vul_arg_parse(argc, argv);
    
    if(rank == 0) {
        vul_arg_display_usage("No Warning, just echoing");
    }

    const unsigned int FILTER_PARAMS = 12; //11 normal plus drift
    const unsigned int BASICPARAMS = 7;
    const unsigned int STATICPARAMS = 2;
    const unsigned int RETRIES = 3;
    const unsigned int ERASE = 2;

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);
    fprintf(stderr, "Brainid Version: %s\n", BRAINID_VERSION);

    Image4DType::Pointer inImage;
    Image4DType::Pointer paramMuImg;
    Image4DType::Pointer paramVarImg;

    std::vector<Activation> input;

    Image3DType::Pointer rms;
    Label4DType::Pointer mask;
    Image4DType::SizeType outsize;

    ofstream ofile("/dev/null");
    ostream* output;
    if(rank == 0) {
        output = &cout;
    } else {
        output = &ofile;
    }

    /* Open up the input */
    try {
        ImageReaderType::Pointer reader;
        reader = ImageReaderType::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_input() );
        reader->Update();
        inImage = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_input().c_str());
        exit(-1);
    }

    if(!a_mask().empty()) try{
        itk::ImageFileReader<Label4DType>::Pointer reader;
        reader = itk::ImageFileReader<Label4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_mask() );
        reader->Update();
        mask = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_mask().c_str());
        exit(-2);
    }
    
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
    *output << "Creating Output Images" << endl;
//    for(int i = 0 ; i < 3 ; i++)
//        outsize[i] = inImage->GetRequestedRegion().GetSize()[i];
    outsize = inImage->GetRequestedRegion().GetSize();
    outsize[3] = BASICPARAMS + STATICPARAMS;
    
    paramMuImg = Image4DType::New();
    paramMuImg->SetRegions(outsize);
    paramMuImg->Allocate();
    paramMuImg->FillBuffer(0);
    
    paramVarImg = Image4DType::New();
    paramVarImg->SetRegions(outsize);
    paramVarImg->Allocate();
    paramVarImg->FillBuffer(0);
    
    
    unsigned int xlen = inImage->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = inImage->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = inImage->GetRequestedRegion().GetSize()[2];
    unsigned int tlen = inImage->GetRequestedRegion().GetSize()[3];

    inImage = preprocess_help(inImage, input, a_timestep(), ERASE, a_delta(),
                a_smart());

    //acquire rms
    rms = get_rms(inImage);
    
    unsigned int method = BoldPF::DIRECT;
    if(a_delta()) method = BoldPF::DELTA;
    else if(a_dc()) method = BoldPF::DC;

    //callback variables, to fill in 
    BoldPF::CallPoints callpoints;
    void* cbdata = NULL;
    int (*cbfunc)(BoldPF*, void*) = NULL;
    if(a_level() == 0) {
        cb_meas_data* cbd = new cb_meas_data;
        cb_meas_init(cbd, &callpoints, inImage->GetRequestedRegion().GetSize());
        cbdata = (void*)cbd;
        cbfunc = cb_meas_call;
    } else {
        cb_part_data* cbd = new cb_part_data;
        cb_part_init(cbd, &callpoints, FILTER_PARAMS, a_num_particles(), 
                    inImage->GetRequestedRegion().GetSize()[3]);
        cbdata = (void*)cbd;
        cbfunc = cb_part_call;

    }
    
    /* Temporary variables used in the loop */
    time_t start = time(NULL);
    Image3DType::IndexType index3 = {{0, 0, 0}};
    Image4DType::PointType point4;
    Image4DType::IndexType index4 = {{0, 0, 0, 0}};
    int result;
    
    aux::vector mu;
    aux::vector var;
    std::vector< aux::vector > meas(tlen, aux::zero_vector(1));

    /* Set constant A1, A2 */
    aux::vector a_values(2);
    a_values[0] = BoldModel::getA1();
    a_values[1] = BoldModel::getA2();
       
    /* Calculate parameters for every voxel */
    for(index3[0] = 0 ; index3[0] < xlen ; index3[0]++) {
        for(index3[1] = 0 ; index3[1] < ylen ; index3[1]++) {
            for(index3[2] = 0 ; index3[2] < zlen ; index3[2]++) {
                //initialize some indexes
                for(int i = 0 ; i < 3 ; i++) index4[i] = index3[i];
                index4[3] = 0;
                inImage->TransformIndexToPhysicalPoint(index4, point4);

                result = BoldPF::UNSTARTED;

                //debug
                *output << index3 << endl;
                *output << index3[0]*ylen*zlen + (index3[2]+1)+index3[1]*zlen << "/" 
                            << xlen*ylen*zlen << endl;

                //run particle filter, and retry with i times as many particles
                //as the the initial number if it fails
                for(unsigned int i = 0 ; checkmask(mask, point4) && 
                            result != BoldPF::DONE && i < RETRIES; i++) { 
                    *output << "RESTARTING!!!!\n" ;
                    fillvector(meas, inImage, index4, a_delta());

                    //create the bold particle filter
                    BoldPF boldpf(meas, input, rms->GetPixel(index3)/5., a_timestep(),
                            output, a_num_particles()*(1<<i), 1./a_divider(), method,
                            a_expweight());
                    
                    //set the callback function
                    for(unsigned int j = 0; j < 3 ; j++)
                        ((cb_data*)cbdata)->pos[j] = index4[j];
                    boldpf.setCallBack(callpoints, cbfunc);

                    //run the particle filter
                    result = boldpf.run(cbdata);
                    mu = boldpf.getDistribution().getDistributedExpectation();
                    aux::matrix cov = boldpf.getDistribution().getDistributedCovariance();
                    var = diag(cov);
                
                }

                //set the output to a standard -1 if BoldPF failed
                if(result != BoldPF::DONE) {
                    mu = aux::vector(BASICPARAMS+STATICPARAMS, -1);
                    var = aux::vector(BASICPARAMS+STATICPARAMS, -1);
                }
                //write the calculated expected value/variance of parameters
                writeVector<double, aux::vector>(paramMuImg, 3, mu, index4);
                writeVector<double, aux::vector>(paramVarImg, 3, var, index4);

                //write a_1 and a_2
                index4[3] = BASICPARAMS;
                writeVector<double, aux::vector>(paramMuImg, 3, a_values, index4);
                writeVector<double, aux::vector>(paramVarImg, 3, aux::vector(2,0),
                            index4);
            
                //run time calculation
                time_t tmp = time(NULL);
                double traveled = (index3[0]*ylen + index3[1])*zlen+index3[2];
                double total = xlen*ylen*zlen;
                cerr << "Time Elapsed: " << difftime(tmp, start) << endl
                     << "Time Remaining: " << (total-traveled)*difftime(tmp,start)/traveled 
                     << endl << "Ratio: " << traveled << "/" << total << endl
                     << "Left: " << total-traveled << "/" << total
                     << endl;
            }
        }
    }
    
    //write final output
    if(rank == 0) {
        paramMuImg->CopyInformation(inImage);
        itk::ImageFileWriter<Image4DType>::Pointer out1 = 
                    itk::ImageFileWriter<Image4DType>::New();
        out1->SetInput(paramMuImg);
        out1->SetFileName("param_exp.nii.gz");
        cout << "Writing: param_exp.nii.gz" << endl;
        out1->Update();

        paramVarImg->CopyInformation(inImage);
        itk::ImageFileWriter<Image4DType>::Pointer out2 = 
                    itk::ImageFileWriter<Image4DType>::New();
        out2->SetInput(paramVarImg);
        out2->SetFileName("param_var.nii.gz");
        cout << "Writing: param_var.nii.gz" << endl;
        out2->Update();
        
        std::ostringstream oss("");
        //write final position or measurement image, 
        if(a_level() == 1) {
            cb_part_data* cdata  = (cb_part_data*)cbdata;
            for(int i = 0 ; i < 3 ; i++)
                oss << cdata->prev[i] << "_";
            oss << ".nii";
        } else {
            oss << "meas_mu.nii.gz";
        }

        itk::ImageFileWriter<itk::OrientedImage<float, 4> >::Pointer writer3 = 
                    itk::ImageFileWriter<itk::OrientedImage<float, 4> >::New();
        writer3->SetFileName(oss.str());
        writer3->SetInput(((cb_data*)cbdata)->image);
        writer3->Update();
        cout << "Writing " << oss.str();
    }

                
    return 0;

}


