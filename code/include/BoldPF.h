#ifndef BOLDPF_H
#define BOLDPF_H
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
#include "RegularizedParticleResamplerMod.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <string>

namespace aux = indii::ml::aux;
typedef indii::ml::filter::ParticleFilter<double> Filter;

/* 
 * YITER - Measurement iterator, over list of doubles/ints/numbers
 * UITER - Stimulus iterator, over list of activations
 * VOID  - Some type we don't care about
 * callback - a callback function that returns a status code
*/
template <typename VOID, int callback(typename const BoldPF*, VOID*)>
class BoldPF 
{
public:

    //SHOULD ONLY BE USED FOR NON-HEAD NODES (!0)
    BoldPF()
    {
        boost::mpi::communicator world;
        const unsigned int rank = world.rank();
        const unsigned int size = world.size();
        assert(rank != 0);

        double tmpvar = weightvar;
        size_t tmppart = particles; 

        boost::mpi::broadcast(world, dt_l, 0);
        boost::mpi::broadcast(world, dt_s, 0);
        boost::mpi::broadcast(world, tmpvar, 0);
        boost::mpi::broadcast(world, tmppart, 0);

        /* Initalized the filter*/
        aux::vector tmp_rms(1);
        tmp_rms[0] = weightvar;
    
        model = new BoldModel(tmp_rms);
        aux::DiracMixturePdf tmp(model.getStateSize());
        filter = new Filter(model, tmp);

        dt_l = longstep;
        dt_s = shortstep;

        read_end = readend;
        stim_end = uend;
    };

    /*SHOULD ONLY BE USED FOR MAIN NODE (0)
     * measurements - a standard vector of measurement aux::vectors
     * activations  - a std::vector of Activation structs, time/level pairs
     * weightvar    - variance of the weighting function's pdf
     * longstep     - amount of time between measurements
     * numparticles - number of particles to use
     * shortstep    - simulation timesteps
     */
    BoldPF(const std::vector<aux::vector>& measurements, 
                const std::vector<Tuple> activations&,  double weightvar,
                double longstep, size_t numparticles = 1000, double shortstep = 1./64)
    {
        boost::mpi::communicator world;
        const unsigned int rank = world.rank();
        const unsigned int size = world.size();
        assert(rank == 0);
        
        dt_l = longstep;
        dt_s = shortstep;

        double tmpvar = weightvar;
        size_t tmppart = particles; 

        boost::mpi::broadcast(world, dt_l, 0);
        boost::mpi::broadcast(world, dt_s, 0);
        boost::mpi::broadcast(world, tmpvar, 0);
        boost::mpi::broadcast(world, tmppart, 0);

        /* Initalized the filter*/
        aux::vector tmp_rms(1) = {{weightvar}};
        model = new BoldModel(tmp_rms);
        aux::DiracMixturePdf tmp(model.getStateSize());
        filter = new Filter(model, tmp);
        
        read_end = readend;
        stim_end = uend;

    };

    int run(YITER ystart, UITER ustart, T* pass);

    void setNumParticles(int newnum);
    int getNumParticles();



private:
    /* Variables */
    //Identity
    Filter* filter;
    BoldModel* model;
    
    //timestep data
    double dt_l;
    double dt_s;

    /* Gathers all the elements of the DiracMixturePdf to the local node */
    void gatherToNode(unsigned int dest, aux::DiracMixturePdf& input) 
    {
      boost::mpi::communicator world;
      unsigned int rank = world.rank();
      unsigned int size = world.size();
      
      assert(dest < size);
    
      std::vector< std::vector< DiracPdf > > xsFull;
      std::vector< aux::vector > wsFull;
    
      unsigned int initialSize = input.getDistributedSize();
      aux::vector initialMu = input.getDistributedExpectation();
      aux::matrix initialCov = input.getDistributedCovariance();
    
      /* if rank is the destination then receive from all the other nodes */
      if(rank == dest) {
        /* Receive from each other node */
        boost::mpi::gather(world, input.getAll(), xsFull, dest); 
        boost::mpi::gather(world, input.getWeights(), wsFull, dest); 
    
        for(unsigned int ii=0 ; ii < size ; ii++) {
          if(ii != rank) {
            for (unsigned int jj = 0; jj < xsFull[ii].size(); jj++) {
              input.add( (xsFull[ii])[jj] , (wsFull[ii])(jj) );
            }
          }
        }
      
      /* if rank is not the destination then send to the destination */
      } else {
        boost::mpi::gather(world, input.getAll(), dest); 
        boost::mpi::gather(world, input.getWeights(), dest); 
        input.clear();
      }
      
      unsigned int endSize = input.getDistributedSize();
      aux::vector endMu = input.getDistributedExpectation();
      aux::matrix endCov = input.getDistributedCovariance();
      
      assert (initialSize == endSize);
    };

}

//*S* means it must be sync'd between all mpi processes
//model         *S* - the input model with the appropriate weighting function etc
//particles     *S* - TOTAL number of particles to use
//longstep      *S* - time between samples from measurement image
//shortstep     *S* - timesteps to simulate at
//timeseries_img    - Image with measurements in it
//index             - Location to read from 
//input_v           - Input/Stimulus vector
template < typename T, int callback(aux::DiractMixturePdf*, T*) >
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

#endif //BOLDPF_H
