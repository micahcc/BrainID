//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan
#include "version.h"

#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkMetaDataObject.h>

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

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;

namespace aux = indii::ml::aux;
typedef indii::ml::filter::ParticleFilter<double> Filter;

void fillvector(std::vector< aux::vector >& output, Image4DType* input,
            Image4DType::IndexType pos)
{       
    ImgIter iter(input, input->GetRequestedRegion());
    iter.SetDirection(3);
    iter.SetIndex(pos);

    output.resize(input->GetRequestedRegion().GetSize()[3]);
    int i = 0;
    while(!iter.IsAtEndOfLine()) {
        output[i] = aux::vector(1, iter.Get());
        ++iter;
        i++;
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

    Image4DType::Pointer inImage;
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

    //detrend, find percent difference, remove the 2 times (since they are typically
    // polluted    
    *output << "Conditioning FMRI Image" << endl;
    inImage = conditionFMRI(inImage, 20.0, input, a_timestep(), 2);
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
    
    for(unsigned int xx = 0 ; xx < xlen ; xx++) {
        for(unsigned int yy = 0 ; yy < ylen ; yy++) {
            for(unsigned int zz = 0 ; zz < zlen ; zz++) {
                //initialize some variables
                Image3DType::IndexType index3 = {{xx, yy, zz}};
                Image4DType::IndexType index4 = {{xx, yy, zz, 0}};
                int result = 0;
                aux::vector mu;
                aux::vector var;
                aux::vector a_values(2);

                //debug
                *output << xx << " " << yy << " " << zz << endl;
                *output << xx*ylen*zlen + (zz+1)+yy*zlen << "/" << xlen*ylen*zlen << endl;

                //run particle filter, and retry with i times as many particles
                //as the the initial number if it fails
                for(int i = 1 ; tmeanImg->GetPixel(index3) > 10 && 
                            result != BoldPF::DONE && i <= RETRIES; i++) { 
                    std::vector< aux::vector > meas(tlen);
                    fillvector(meas, inImage, index4);

                    BoldPF boldpf(meas, input, rms->GetPixel(index3), a_timestep(),
                            output, a_num_particles()*i, 1./a_divider());
                    result = boldpf.run();
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
                writeVector<double, aux::vector>(paramVarImg, 3, aux::vector(2,0), index4);
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
    }
                
    return 0;

}


