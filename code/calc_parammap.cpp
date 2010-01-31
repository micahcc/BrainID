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
    std::cerr << "Copied " << i << " doubles into vector" << std::endl;
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

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    Image4DType::Pointer inImage;
    Image4DType::Pointer meanImage;
    Image4DType::Pointer varImage;

    std::vector<Activation> input;

    Image3DType::Pointer rms;
    Label3DType::Pointer mask;

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
    if(rank == 0) {
        fprintf(stdout, "Creating Output Images\n");
        Image4DType::SizeType size;
        for(int i = 0 ; i < 3 ; i++)
            size[i] = inImage->GetRequestedRegion().GetSize()[i];
        size[3] = 7;
        
        meanImage = Image4DType::New();
        meanImage->SetRegions(size);
        meanImage->Allocate();
        meanImage->FillBuffer(0);
        
        varImage = Image4DType::New();
        varImage->SetRegions(size);
        varImage->Allocate();
        varImage->FillBuffer(0);
    }
    
    unsigned int xlen = inImage->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = inImage->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = inImage->GetRequestedRegion().GetSize()[2];
    unsigned int tlen = inImage->GetRequestedRegion().GetSize()[3];
    //Find the Tmean, and ignore elemnts whose mean is < 1
    Image3DType::Pointer mean = Tmean(inImage);
    if(rank == 0) {
        itk::ImageFileWriter<Image3DType>::Pointer out = 
                    itk::ImageFileWriter<Image3DType>::New();
        out->SetInput(mean);
        string tmean = a_output();
        out->SetFileName(tmean.append("/Tmean.nii.gz"));
        cout << "Writing: " << tmean << endl;
        out->Update();
    }

    //detrend, find percent difference, remove the 2 times (since they are typically
    // polluted    
    ofstream ofile("/dev/null");
    ostream* output;
    if(rank == 0) {
        output = &cout;
        *output << "Conditioning FMRI Image" << endl;
    } else {
        output = &ofile;
    }
    inImage = conditionFMRI(inImage, 20.0, input, a_timestep(), 2);
    /* Save detrended image */
    if(rank == 0) {
        itk::ImageFileWriter<Image4DType>::Pointer out = 
                    itk::ImageFileWriter<Image4DType>::New();
        out->SetInput(inImage);
        string pfilter = a_output();
        out->SetFileName(pfilter.append("/pfilter_input.nii.gz"));
        cout << "Writing: " << pfilter << endl;
        out->Update();
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
                aux::vector mu(meanImage->GetRequestedRegion().GetSize()[3]);
                aux::vector var(meanImage->GetRequestedRegion().GetSize()[3]);
                
                //debug
                *output << xx << " " << yy << " " << zz << endl;
                *output << xx*ylen*zlen + (zz+1)+yy*zlen << "/" << xlen*ylen*zlen << endl;
                
                //run particle filter
                if(mean->GetPixel(index3) > 10) {
                    std::vector< aux::vector > meas(tlen);
                    fillvector(meas, inImage, index4);
                    
                    BoldPF boldpf(meas, input, rms->GetPixel(index3), a_timestep(),
                                output, a_num_particles(), 1./a_divider());
                    result = boldpf.run();
                    mu = boldpf.getDistribution().getDistributedExpectation();
                    aux::matrix cov = boldpf.getDistribution().getDistributedCovariance();
                    var = diag(cov);
                }

                //write the output
                if(rank == 0) {
                    if(result != 3) {
                        for(unsigned int i = 0 ; i < mu.size() ; i++) 
                            mu[i] = -1;
                        for(unsigned int i = 0 ; i < var.size() ; i++)
                            var[i] = -1;
                    }
                    writeVector<double, aux::vector>(meanImage, 3, mu, index4);
                    writeVector<double, aux::vector>(varImage, 3, var, index4);
                }
            }
        }
    }
                
  return 0;

}


