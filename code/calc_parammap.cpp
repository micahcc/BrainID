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

struct callback_data
{
    Image4DType::Pointer image;
    Image4DType::IndexType pos;
};

int callback(BoldPF* bold, void* data)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
//    const unsigned int size = world.size();
    
    callback_data* cdata = (struct callback_data*)data;
    cdata->pos[3] = bold->getDiscTimeL();
    aux::vector meas =  bold->getModel().measure(
                bold->getDistribution().getDistributedExpectation());
    if(rank == 0) 
         cdata->image->SetPixel(cdata->pos, meas[0]);
    std::cout << "time: " << bold->getContTime() << "\n";
    return 0;
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
            

/* Main Function */
int main(int argc, char* argv[])
{
    /* Initialize mpi */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    vul_arg<string> a_input(0, "4D timeseries file");
    vul_arg<string> a_output(0, "output directory");
    
    vul_arg<bool> a_drift("-f", "Add drift term to model.", false);
    vul_arg<bool> a_delta("-l", "Use deltas between measurements, this precludes"
                "the drift option", false);
    
    vul_arg<string> a_mask("-m", "3D mask file");
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

    const unsigned int BASICPARAMS = 7;
    const unsigned int STATICPARAMS = 2;
    const unsigned int RETRIES = 3;

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);
    fprintf(stderr, "Brainid Version: %s\n", BRAINID_VERSION);

    Image4DType::Pointer inImage;
    Image4DType::Pointer measMuImg;
    Image4DType::Pointer paramMuImg;
    Image4DType::Pointer paramVarImg;

    std::vector<Activation> input;

    Image3DType::Pointer rms;
    Label3DType::Pointer mask;
    Image4DType::SizeType outsize;

    string tmp;
    
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

    try{
    if(!a_mask().empty()) {
        itk::ImageFileReader<Label3DType>::Pointer reader;
        reader = itk::ImageFileReader<Label3DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_mask() );
        reader->Update();
        mask = reader->GetOutput();
    }
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
    for(int i = 0 ; i < 3 ; i++)
        outsize[i] = inImage->GetRequestedRegion().GetSize()[i];
    outsize[3] = BASICPARAMS + STATICPARAMS;
    
    paramMuImg = Image4DType::New();
    paramMuImg->SetRegions(outsize);
    paramMuImg->Allocate();
    paramMuImg->FillBuffer(0);
    
    paramVarImg = Image4DType::New();
    paramVarImg->SetRegions(outsize);
    paramVarImg->Allocate();
    paramVarImg->FillBuffer(0);
    
    measMuImg = Image4DType::New();
    measMuImg->SetRegions(inImage->GetRequestedRegion());
    measMuImg->Allocate();
    measMuImg->FillBuffer(0);
    
    unsigned int xlen = inImage->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = inImage->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = inImage->GetRequestedRegion().GetSize()[2];
    unsigned int tlen = inImage->GetRequestedRegion().GetSize()[3];
    //Find the Tmean, and ignore elemnts whose mean is < 1
    Image3DType::Pointer tmeanImg = Tmean(inImage);
    if(rank == 0) try {
        itk::ImageFileWriter<Image3DType>::Pointer out = 
                    itk::ImageFileWriter<Image3DType>::New();
        out->SetInput(tmeanImg);
        tmp = a_output();
        out->SetFileName(tmp.append("/Tmean.nii.gz"));
        cout << "Writing: " << tmp << endl;
        out->Update();
    } catch(itk::ExceptionObject) {
        cerr << "Error opening " << tmp << endl;
        exit(-3);
    }

    /* Set up measurements image */
    *output << "Conditioning FMRI Image" << endl;
    //remove first 2 time step, since they are typically polluted
    inImage = pruneFMRI(inImage, input, a_timestep(), 2);
    unsigned int method;

    //calculate %difference, which is used normally for the bold signal
    // or the modified % difference (with spline rather than mean)
    if(a_delta() || a_drift()) {
        *output << "Changing Image to %difference" << endl;
        method = a_delta() ? BoldPF::DELTA : BoldPF::PROCESS;
        SubF4::Pointer sub = SubF4::New();   
        DivF4::Pointer div = DivF4::New();
        Image4DType::Pointer mean = extrude(Tmean(inImage),
                    inImage->GetRequestedRegion().GetSize()[3]);
        sub->SetInput1(inImage);
        sub->SetInput2(mean);
        div->SetInput1(sub->GetOutput());
        div->SetInput2(mean);
        div->Update();
        inImage = div->GetOutput();
    } else {
        *output << "De-trending, then dividing by mean" << endl;
        method = BoldPF::DIRECT;
        inImage = deSpline(inImage, 6, input, a_timestep());
    }

    /* Save detrended image */
    if(rank == 0) try {
        itk::ImageFileWriter<Image4DType>::Pointer out = 
                    itk::ImageFileWriter<Image4DType>::New();
        out->SetInput(inImage);
        tmp = a_output();
        out->SetFileName(tmp.append("/pfilter_input.nii.gz"));
        cout << "Writing: " << tmp << endl;
        out->Update();
    } catch(itk::ExceptionObject) {
        cerr << "Error opening " << tmp << endl;
        exit(-4);
    }
    
    //acquire rms
    rms = get_rms(inImage);

    //callback variables, to fill in 
    BoldPF::CallPoints callpoints;
    callpoints.start = false;
    callpoints.postMeas = true;
    callpoints.postFilter = false;
    callpoints.end = false;

    callback_data cbd;
    cbd.image = measMuImg;
    
    time_t start = time(NULL);
    for(unsigned int xx = 0 ; xx < xlen ; xx++) {
        for(unsigned int yy = 0 ; yy < ylen ; yy++) {
            for(unsigned int zz = 0 ; zz < zlen ; zz++) {
                //initialize some variables
                Image3DType::IndexType index3 = {{xx, yy, zz}};
                Image4DType::IndexType index4 = {{xx, yy, zz, 0}};
                cbd.pos = index4;
                int result = 0;
                aux::vector mu;
                aux::vector var;
                aux::vector a_values(2);

                //debug
                *output << xx << " " << yy << " " << zz << endl;
                *output << xx*ylen*zlen + (zz+1)+yy*zlen << "/" << xlen*ylen*zlen 
                            << endl;

                //run particle filter, and retry with i times as many particles
                //as the the initial number if it fails
                for(unsigned int i = 0 ; tmeanImg->GetPixel(index3) > 10 && 
                            result != BoldPF::DONE && i < RETRIES; i++) { 
                    std::vector< aux::vector > meas(tlen);
                    fillvector(meas, inImage, index4, a_delta());

                    BoldPF boldpf(meas, input, rms->GetPixel(index3), a_timestep(),
                            &ofile, a_num_particles()*(1<<i), 1./a_divider(), method);
                    boldpf.setCallBack(callpoints, &callback);
                    result = boldpf.run(&cbd);
                    mu = boldpf.getDistribution().getDistributedExpectation();
                    aux::matrix cov = boldpf.getDistribution().getDistributedCovariance();
                    var = diag(cov);
                
                    a_values[0] = boldpf.getModel().getA1();
                    a_values[1] = boldpf.getModel().getA2();
                }

                //save the output
                if(result != BoldPF::DONE) {
                    mu = aux::vector(BASICPARAMS, -1);
                    var = aux::vector(BASICPARAMS, -1);
                }
                //write the calculated expected value/variance of parameters
                writeVector<double, aux::vector>(paramMuImg, 3, mu, index4);
                writeVector<double, aux::vector>(paramVarImg, 3, var, index4);

                //write a_1 and a_2
                index4[3] = BASICPARAMS;
                writeVector<double, aux::vector>(paramMuImg, 3, a_values, index4);
                writeVector<double, aux::vector>(paramVarImg, 3, aux::vector(2,0),
                            index4);
            
                time_t tmp = time(NULL);
                double traveled = xx*yy*zz+zlen*yy+zz;
                double total = xlen*ylen*zlen;
                cerr << "Elapsed: " << difftime(tmp, start) << endl
                     << "Remaining: " << (total-traveled)*difftime(tmp,start)/traveled
                     << endl;
            }
        }
    }
    
    //write final output
    if(rank == 0) {
        itk::ImageFileWriter<Image4DType>::Pointer out1 = 
                    itk::ImageFileWriter<Image4DType>::New();
        out1->SetInput(paramMuImg);
        string tmp1 = a_output();
        out1->SetFileName(tmp1.append("/param_exp.nii.gz"));
        cout << "Writing: " << tmp1 << endl;
        out1->Update();

        itk::ImageFileWriter<Image4DType>::Pointer out2 = 
                    itk::ImageFileWriter<Image4DType>::New();
        out2->SetInput(paramVarImg);
        string tmp2 = a_output();
        out2->SetFileName(tmp2.append("/param_var.nii.gz"));
        cout << "Writing: " << tmp2 << endl;
        out2->Update();
        
        itk::ImageFileWriter<Image4DType>::Pointer out3 = 
                    itk::ImageFileWriter<Image4DType>::New();
        out3->SetInput(measMuImg);
        string tmp3 = a_output();
        out3->SetFileName(tmp3.append("/meas_mu.nii.gz"));
        cout << "Writing: " << tmp3 << endl;
        out3->Update();
    }
                
    return 0;

}


