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

int callback(const BoldPF<double>* in, double garbage)
{
    return 1;
};

void fillvector(std::vector< aux::vector > output, Image4DType* input,
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
    vul_arg<string> a_mask(0, "3D mask file");
    vul_arg<string> a_output(0, "output directory");
    
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
    Image4DType::Pointer outImage;

    std::vector<Activation> input;

    Image3DType::Pointer rms;
    Label3DType::Pointer mask;

    /* Open up the input */
    {
    ImageReaderType::Pointer reader;
    reader = ImageReaderType::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_input() );
    reader->Update();
    inImage = reader->GetOutput();
    }
    {
    itk::ImageFileReader<Label3DType>::Pointer reader;
    reader = itk::ImageFileReader<Label3DType>::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    reader->SetFileName( a_input() );
    reader->Update();
    mask = reader->GetOutput();
    }
    
    /* Open Stimulus file */
    input = read_activations(a_stimfile().c_str());

    /* Create Output Image */
    if(rank == 0) {
        outImage = Image4DType::New();
        Image4DType::SizeType size;
        for(int i = 0 ; i < 3 ; i++)
            size[i] = inImage->GetRequestedRegion().GetSize()[i];
        size[3] = 7;

        outImage->SetRegions(size);
        outImage->Allocate();
    }
    
    unsigned int xlen = inImage->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = inImage->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = inImage->GetRequestedRegion().GetSize()[2];
    unsigned int tlen = inImage->GetRequestedRegion().GetSize()[3];
    
    //detrend
    inImage = normalizeByVoxel(inImage, mask, 
                inImage->GetRequestedRegion().GetSize()[3]/10+2);
    /* Save detrended image */
    if(rank == 0) {
        itk::ImageFileWriter<Image4DType>::Pointer out = 
                    itk::ImageFileWriter<Image4DType>::New();
        out->SetInput(inImage);
        out->SetFileName("pfilter_input.nii.gz");
        out->Update();
    }
    
    //acquire rms
    rms = get_rms(inImage);

    for(unsigned int xx = 0 ; xx < xlen ; xx++) {
        for(unsigned int yy = 0 ; yy < ylen ; yy++) {
            for(unsigned int zz = 0 ; zz < zlen ; zz++) {
                Image3DType::IndexType index3 = {{xx, yy, zz}};
                Image4DType::IndexType index4 = {{xx, yy, zz, 0}};
                std::vector< aux::vector > meas(tlen);
                fillvector(meas, inImage, index4);
                
                BoldPF<int> boldpf(meas, input, rms->GetPixel(index3), a_timestep(),
                            1./a_divider(), a_num_particles());
                boldpf.run();
            }
        }
    }
                
  return 0;

}


