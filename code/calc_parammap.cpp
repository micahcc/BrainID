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

#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
#include <indii/ml/aux/DiracMixturePdf.hpp>
#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <indii/ml/aux/Almost2Norm.hpp>
#include <indii/ml/aux/AlmostGaussianKernel.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "tools.h"
#include "segment.h"
#include "modNiftiImageIO.h"
#include "BoldModel.hpp"
#include "RegularizedParticleResamplerMod.hpp"
//#include "BoldPF.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;

namespace aux = indii::ml::aux;
typedef indii::ml::filter::ParticleFilter<double> Filter;

int callback(const BoldPF* in, double garbage)
{
    return 1;
}

/* Gathers all the elements of the DiracMixturePdf to the local node */
//void gatherToNode(unsigned int dest, aux::DiracMixturePdf& input) 
//{
//  boost::mpi::communicator world;
//  unsigned int rank = world.rank();
//  unsigned int size = world.size();
//  
//  assert(dest < size);
//
//  std::vector< std::vector< DiracPdf > > xsFull;
//  std::vector< aux::vector > wsFull;
//
//  unsigned int initialSize = input.getDistributedSize();
//  aux::vector initialMu = input.getDistributedExpectation();
//  aux::matrix initialCov = input.getDistributedCovariance();
//
//  /* if rank is the destination then receive from all the other nodes */
//  if(rank == dest) {
//    /* Receive from each other node */
//    boost::mpi::gather(world, input.getAll(), xsFull, dest); 
//    boost::mpi::gather(world, input.getWeights(), wsFull, dest); 
//
//    for(unsigned int ii=0 ; ii < size ; ii++) {
//      if(ii != rank) {
//        for (unsigned int jj = 0; jj < xsFull[ii].size(); jj++) {
//          input.add( (xsFull[ii])[jj] , (wsFull[ii])(jj) );
//        }
//      }
//    }
//  
//  /* if rank is not the destination then send to the destination */
//  } else {
//    boost::mpi::gather(world, input.getAll(), dest); 
//    boost::mpi::gather(world, input.getWeights(), dest); 
//    input.clear();
//  }
//  
//  unsigned int endSize = input.getDistributedSize();
//  aux::vector endMu = input.getDistributedExpectation();
//  aux::matrix endCov = input.getDistributedCovariance();
//  
//  assert (initialSize == endSize);
//};

//*S* means it must be sync'd between all mpi processes
//model         *S* - the input model with the appropriate weighting function etc
//particles     *S* - TOTAL number of particles to use
//longstep      *S* - time between samples from measurement image
//shortstep     *S* - timesteps to simulate at
//timeseries_img    - Image with measurements in it
//index             - Location to read from 
//input_v           - Input/Stimulus vector
template < typename T, int callback(typename aux::DiracMixturePdf*, T*) >
int calcParams(Filter* filter, size_t particles, double longstep, double shortstep,
            Image4DType::Pointer timeseries_img, Image3DType::IndexType index,
            std::vector<Tuple> input_v, T* pass)
{
    /* Initialize mpi */
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    /* GetSize of Input Image */
    size_t tlength;
    if(rank == 0) {
        tlength = timeseries_img->GetRequestedRegion().GetSize()[3]; 
    }
    boost::mpi::broadcast(world, tlength, 0);
    Image4DType::IndexType pos = {{index[0], index[1], index[2], 0}};

    ostream* out;
    ofstream nullout("/dev/null");
    /* output setup */
    if(rank == 0)
        out = &cout;
    else
        out = &nullout;
    
    /* 
     * Particles Setup 
     */
    *out << "Generating prior" << endl;
    size_t localparticles = particles/size;
    
    //give excess particles to last rank
    if(rank == (size-1))
        localparticles += particles - localparticles*size;
    filter->getModel()->generatePrior(filter->getFilteredState(), 
                localparticles, 4); //2*sigma, squared

    //Redistribute - doesn't cost anything if distrib. was already fine 
    *out << "Redistributing" << endl;
    filter->getFilteredState().redistributeBySize(); 

    *out << "Size: " <<  filter->getFilteredState().getSize() << endl;
    
    /* 
     * Create resamplers 
     *
     * Normal resampler, used to eliminate particles 
     */
    indii::ml::filter::StratifiedParticleResampler resampler;

    /* Regularized Resample */
    aux::Almost2Norm norm;
    aux::AlmostGaussianKernel kernel(filter->getModel()->getStateSize(), 1);
    RegularizedParticleResamplerMod<aux::Almost2Norm, aux::AlmostGaussianKernel> 
                resampler_reg(norm, kernel, filter->getModel());

    /* Simulation Section */
    aux::vector input(1);
    input[0] = 0;
    aux::vector meas(1);
    meas[0] = 0;
    
    double conttime = 0;
    size_t stim_index = 0;
    
    /* 
     * Run the particle filter either until we reach a predetermined end
     * time, or until we are done processing measurements.
     */
     for(int disctime = 0; disctime*shortstep < longstep*tlength ; disctime++) {
        /* time */
        conttime = disctime*shortstep;
        *out << "t= " << conttime << ", ";
        
        /* Update Input if there is any*/
        while(rank == 0 && input_v[stim_index].time <= conttime) {
            input[0] = input_v[stim_index].level;
            stim_index++;
        }
        
        boost::mpi::broadcast(world, input, 0);
        model.setinput(input);

        /* Check to see if it is time to update */
        if(conttime >= pos[3]*longstep) { 
            //acquire the latest measurement
            if(rank == 0) {
                *out << "Measuring at " <<  conttime << endl;
                meas[0] = timeseries_img->GetPixel(pos);
                outputVector(*out, meas);
                *out << endl;
            }

            //send meas and done to other nodes
            boost::mpi::broadcast(world, meas, 0);
            
            //step forward in time, with measurement
            filter->filter(conttime, meas);

            //check to see if resampling is necessary
            double ess = filter->getFilteredState().calculateDistributedEss();
            
            //check for errors, could be caused by total collapse of particles,
            //for instance if all the particles go to an unreasonable value like 
            //inf/nan/neg
            if(isnan(ess) || isinf(ess)) {
                *out << std::endl << "Error! ESS was " << ess << endl;
                return -1;
            } 
            
            /* Because of the weighting functions, sometimes the total weight
             * can get extremely high, thus this drops it back down if the
             * total weight gets too high 
             */
            double totalweight = filter->getFilteredState().getDistributedTotalWeight();
            *out << "Total Weight: " << totalweight << endl;

            //time to resample
            if(ess < 50) {
                *out << " ESS: " << ess << ", Stratified Resampling" << endl;
                aux::symmetric_matrix statecov;
                statecov = filter->getFilteredState().getDistributedCovariance();
                filter->resample(&resampler);
                
                *out << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter->setFilteredState(resampler_reg.
                            resample(filter->getFilteredState(), statecov) );
            } else {
                *out << " ESS: " << ess << ", No Resampling Necessary!" 
                            << endl;
            }

            pos[3]++;
        } else { //no update available, just step update states
            filter->filter(conttime);
        }
   
       
        /* Update Time, disctime */
        disctime++;
    }

    *out << "End time: "<< conttime << endl;
    
    return 0;
}

/* Main Function */
int main(int argc, char* argv[])
{
    /* Initialize mpi */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    ostream* out;
    ofstream nullout("/dev/null");

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
        out = &cout;
    } else {
        out = &nullout;
    }
        
    if(rank == 0) {
        vul_arg_display_usage("No Warning, just echoing");
    }

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    std::ifstream fin;

    Image4DType::Pointer inImage;
    Image4DType::Pointer outImage;

    //distribute arguments
    double timestep = a_timestep();
    double divider = a_divider();
    double particles = a_num_particles();

    std::vector<Tuple> input;

    size_t xlen, ylen, zlen;
    Image3DType::Pointer rms;
    Label3DType::Pointer mask;

    if(rank == 0) {
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
        *out << a_stimfile() << endl;;
        input = read_activations(a_stimfile().c_str());

        /* Create Output Image */
        outImage = Image4DType::New();
        Image4DType::SizeType size;
        for(int i = 0 ; i < 3 ; i++)
            size[i] = inImage->GetRequestedRegion().GetSize()[i];
        size[3] = 7;

        outImage->SetRegions(size);
        outImage->Allocate();
        
        xlen = inImage->GetRequestedRegion().GetSize()[0];
        ylen = inImage->GetRequestedRegion().GetSize()[1];
        zlen = inImage->GetRequestedRegion().GetSize()[2];
        
        //detrend
        Image4DType::Pointer tmp = normalizeByVoxel(inImage, mask, 
                    inImage->GetRequestedRegion().GetSize()[3]/10+2);
        {
            itk::ImageFileWriter<Image4DType>::Pointer out = 
                        itk::ImageFileWriter<Image4DType>::New();
            out->SetInput(tmp);
            out->SetFileName("pfilter_input.nii.gz");
            out->Update();
        }
        inImage = tmp;
        //acquire rms
        rms = get_rms(inImage);
    }

    boost::mpi::broadcast(world, xlen, 0);
    boost::mpi::broadcast(world, ylen, 0);
    boost::mpi::broadcast(world, zlen, 0);

    boost::mpi::broadcast(world, timestep, 0);
    boost::mpi::broadcast(world, divider, 0);
    boost::mpi::broadcast(world, particles, 0);

    double tmp_rms;

    for(size_t xx = 0 ; xx < xlen ; xx++) {
        for(size_t yy = 0 ; yy < ylen ; yy++) {
            for(size_t zz = 0 ; zz < zlen ; zz++) {
                //create the model/empty distribution for the filter
                Image4DType::IndexType index = {{xx, yy, zz, 0}};

                if( rank == 0 ) {
                    //Calculate RMS to use for weight functions' variance
                    tmp_rms = rms->GetPixel(index); 

                    //initialize start stop iterators for measurements
                    ImgIter ystart(inImage, inImage->GetRequestedRegion());
                    ystart.SetDirection(3);
                    ystart.SetIndex(index);
                    ImgIter yend = ystart;
                    yend.GoToEndOfLine();

                    //initialize iterators over activations
                    vector<Activation>::iterator ustart = input.begin();
                    vector<Activation>::iterator uend = input.begin();

                    BoldPF<ImgIter, vector<Tuple>, double, callback>
                                boldpf(particles, timestep, 1./divider, 
                                tmp_rms, yend, uend);
                } else {
                    BoldPF<ImgIter, vector<Tuple>, double, callback> boldpf;
                }

                memcpy
                strcpy

                int status = calcParams<(&filter, particles, timestep, 1./divider,
                            inImage, index, input);
                if(status != 0) {
                    *out << "Error at " << xx << "," << yy << "," << zz << endl;
                } else {
                    //get mean from distr

                    //save parameters in output image

                    //todo: save distribution?

                }
                delete filter;
            }
        }
    }
                
  return 0;

}


