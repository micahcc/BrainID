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
#include <fstream>
#include <string>

namespace aux = indii::ml::aux;

/* 
 * VOID  - Some type we don't care about
 * callback - a callback function that returns a status code
*/
class BoldPF 
{
public:
    /* Types */
    typedef indii::ml::filter::ParticleFilter<double> Filter;
    
    /* Callback types */
    struct CallPoints{
        bool start;
        bool postMeas;
        bool postFilter; //AKA when no measurement exists;
        bool end;
    };
    typedef int(*CallBackFunction)(BoldPF*, void*);

    /* Constructor
     * measurements - a standard vector of measurement aux::vectors
     * activations  - a std::vector of Activation structs, time/level pairs
     * weightvar    - variance of the weighting function's pdf
     * longstep     - amount of time between measurements
     * numparticles - number of particles to use
     * shortstep    - simulation timesteps
     */
    BoldPF(const std::vector<aux::vector>& measurements, 
                const std::vector<Activation>& activations,  double weightvar,
                double longstep, unsigned int numparticles = 1000, 
                double shortstep = 1./64);

    /* Destructor */
    ~BoldPF();
    
    /* Primary Functions */
    int run(void* pass);
    int pause() { return status = 2; };

    /* Accessors */
    int getNumParticles() const;
    double getShortStep() const;
    double getLongStep() const;
    aux::DiracMixturePdf& getDistribution();
    int getStatus();

    void setCallBack(const CallPoints& cpt, CallBackFunction cback)
    {
        callback = cback;
        call_points = cpt;
    };

private:
    /* Variables */
    //Identity
    Filter* filter;
    BoldModel* model;
    
    //timestep data
    double dt_l;
    double dt_s;

    //Indices
    unsigned int disctime_l;
    unsigned int disctime_s;

    //0, not started
    //1, started
    //2, started, but paused
    //3, done
    int status;

    //log output
    std::ostream* debug;
    std::ofstream nullout;

    //resamplers
    indii::ml::filter::StratifiedParticleResampler resampler;
    RegularizedParticleResamplerMod<aux::Almost2Norm, aux::AlmostGaussianKernel>*
                resampler_reg;

    //input and measurement vectors
    std::vector<aux::vector> measure;
    std::vector<Activation> stim;

    //constants
    const unsigned int ESS_THRESH;
    
    //Callback data
    CallPoints call_points;
    static int nop(BoldPF*, void*) { return 0; };
    int (*callback)(BoldPF*, void*);

    /* Gathers all the elements of the DiracMixturePdf to the local node */
    void gatherToNode(unsigned int dest, aux::DiracMixturePdf& input);
};
    
    
int BoldPF::getNumParticles() const
{
    return filter->getFilteredState().getDistributedSize();
};

double BoldPF::getShortStep() const
{
    return dt_s;
};

double BoldPF::getLongStep() const
{
    return dt_l;
};

aux::DiracMixturePdf& BoldPF::getDistribution()
{
    return filter->getFilteredState();
};

//const Filter& getFilter()
//{
//    return *filter;
//};

int BoldPF::getStatus()
{
    return status;
};

/* Run - runs the particle filter
 * pass - variable to pass to callback function
**/
int BoldPF::run(void* pass = NULL)
{
    using std::endl;

    /* Initialize mpi */
//    boost::mpi::communicator world;
//    const unsigned int rank = world.rank();
//    const unsigned int size = world.size();
    
    /* Simulation Section */
    double conttime = disctime_s*dt_s;
    unsigned int stim_index = 0;

    *debug << "Starting at " << conttime
                << " disctime_s: " << disctime_s
                << " disctime_l: " << disctime_l 
                << " dt_l: " << dt_l << " dt_s: " << dt_s 
                << " measure size: " << measure.size()
                << " stim size: " << stim.size() << endl;
    
    if(call_points.start) 
        callback(this, pass);
    
    /* 
     * Run the particle filter either until we reach a predetermined end
     * time, or until we are done processing measurements.
     */
     status = 1;
     while(status == 1 && disctime_s*dt_s < dt_l*measure.size()) {
        /* time */
        conttime = disctime_s*dt_s;
        *debug << "."; 
        
        /* Update Input if there is any*/
        while(stim[stim_index].time <= conttime) {
            model->setinput(aux::vector(1, stim[stim_index].level));
            stim_index++;
        }
        

        /* Check to see if it is time to update */
        if(conttime >= disctime_l*dt_l) { 
            //acquire the latest measurement
            *debug << "Measuring at " <<  conttime << endl;
            aux::vector meas(measure[disctime_l]);
            outputVector(*debug, meas);
            *debug << endl;

            //step forward in time, with measurement
            filter->filter(conttime, meas);
            if(call_points.postMeas) 
                callback(this, pass);

            //check to see if resampling is necessary
            double ess = filter->getFilteredState().calculateDistributedEss();
            
            //check for errors, could be caused by total collapse of particles,
            //for instance if all the particles go to an unreasonable value like 
            //inf/nan/neg
            if(isnan(ess) || isinf(ess)) {
                *debug << endl << "Error! ESS was " << ess << endl;
                return -1;
            } 
            
            //time to resample
            if(ess < ESS_THRESH) {
                *debug << " ESS: " << ess << ", Stratified Resampling" << endl;
                aux::symmetric_matrix statecov;
                statecov = filter->getFilteredState().getDistributedCovariance();
                filter->resample(&resampler);
                
                *debug << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter->setFilteredState(
                            resampler_reg->resample(filter->getFilteredState(), statecov));
            } else {
                *debug << " ESS: " << ess << ", No Resampling Necessary!" 
                            << endl;
            }
            
            disctime_l++;
        } else { //no update available, just step update states
            filter->filter(conttime);
            if(call_points.postFilter) 
                callback(this, pass);
        }
   
        /* Update Time, disctime */
        disctime_s++;
    }

    /* Check to see if algorith finished, otherwise it was paused */
    if(disctime_s*dt_s >= dt_l*measure.size()) {
        status = 3;
        if(call_points.end) 
            callback(this, pass);
    }

    *debug << "Stop time: "<< conttime << endl;
    
    return 0;
};

/* Constructor
 * measurements - a standard vector of measurement aux::vectors
 * activations  - a std::vector of Activation structs, time/level pairs
 * weightvar    - variance of the weighting function's pdf
 * longstep     - amount of time between measurements
 * numparticles - number of particles to use
 * shortstep    - simulation timesteps
 */
BoldPF::BoldPF(const std::vector<aux::vector>& measurements, 
            const std::vector<Activation>& activations,  double weightvar,
            double longstep, unsigned int numparticles, double shortstep) : 
            dt_l(longstep), dt_s(shortstep), 
            disctime_l(0), disctime_s(0), status(0), 
            nullout("/dev/null"),
            measure(measurements), stim(activations), 
            ESS_THRESH(50)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    using std::endl;
    
    /* Initalize the model and filter*/
    aux::vector tmp_rms(1, weightvar);
    model = new BoldModel(tmp_rms, false, measurements.front().size());
    aux::vector tmp_in(1,0);
    model->setinput(aux::vector(1, 0));
    aux::DiracMixturePdf tmp(model->getStateSize());
    filter = new indii::ml::filter::ParticleFilter<double>(model, tmp);

    /* initialize debug/output */
    if(rank == 0)
        debug = &std::cout;
    else
        debug = &nullout;

    /* 
     * Particles Setup 
     */
    *debug << "Generating prior" << std::endl;
    unsigned int localparticles = numparticles/size;
    
    //give excess particles to last rank
    if(rank == (size-1))
        localparticles += numparticles - localparticles*size;

    //prior is (2*sigma)^2, by having each rank generate its own particles,
    //a decent amount of time is saved
    model->generatePrior(filter->getFilteredState(), localparticles, 4); 

    //Redistribute - doesn't cost anything if distrib. was already fine 
    *debug << "Redistributing" << endl;
    filter->getFilteredState().redistributeBySize(); 

    *debug << "Size: " <<  filter->getFilteredState().getDistributedSize() << endl;

    /* 
     * Create 
     * Regularized Resampler
     */
    aux::Almost2Norm norm;
    aux::AlmostGaussianKernel kernel(model->getStateSize(), 1);
    resampler_reg = new RegularizedParticleResamplerMod
                <aux::Almost2Norm, aux::AlmostGaussianKernel>
                (norm, kernel, model);

    call_points.start = false;
    call_points.postMeas = false;
    call_points.postFilter = false;
    call_points.end = false;
    callback = nop;
};

/* Destructor */
BoldPF::~BoldPF()
{
    delete filter;
    delete model;
    delete resampler_reg;
};

/* gatherToNode 
 * dest  - destination rank
 * input - mixturePDF whose components will be gathered to dest
**/
void BoldPF::gatherToNode(unsigned int dest, aux::DiracMixturePdf& input)
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

#endif //BOLDPF_H
