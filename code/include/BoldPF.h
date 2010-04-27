#ifndef BOLDPF_H
#define BOLDPF_H
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


#include <indii/ml/filter/ParticleResampler.hpp>
#include <indii/ml/aux/Almost2Norm.hpp>
#include <indii/ml/aux/AlmostGaussianKernel.hpp>
#include "boost/numeric/bindings/traits/ublas_matrix.hpp"
#include "boost/numeric/bindings/traits/ublas_vector.hpp"
#include "boost/numeric/bindings/traits/ublas_symmetric.hpp"
#include "boost/numeric/bindings/lapack/lapack.hpp"

namespace aux = indii::ml::aux;

class BoldPF 
{
public:
    /* Types */
    typedef indii::ml::filter::ParticleFilter<double> Filter;
    enum Status {ERROR=-1, UNSTARTED=0, RUNNING=1, PAUSED=2, DONE=3};
    enum Method {DIRECT, DELTA, DC};
    
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
                double longstep, std::ostream* output, 
                unsigned int numparticles = 1000, double shortstep = 1./64,
                unsigned int method = DIRECT, int weightf = 0, bool flatten = true);

    /* Destructor */
    ~BoldPF();
    
    /* Primary Functions */
    int run(void* pass);
    int pause() { return status = PAUSED; };

    /* Accessors */
    double getContTime() const {return disctime_s*dt_s;};
    unsigned int getDiscTimeL() const {return disctime_l;};
    unsigned int getDiscTimeS() const {return disctime_s;};
    int getNumParticles() const;
    double getShortStep() const;
    double getLongStep() const;
    aux::DiracMixturePdf& getDistribution();
    int getStatus();

    BoldModel& getModel() { return *model; };

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

    //-1, error
    // 0, not started
    // 1, started
    // 2, started, but paused
    // 3, done
    int status;
    int method;

    //log output
    std::ostream* debug;

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
    
    /* Saves in the state variable the previous measurement, for delta */
    void latchBold();
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

int BoldPF::getStatus()
{
    return status;
};

aux::matrix calcCov(indii::ml::aux::DiracMixturePdf& p)
{
    boost::mpi::communicator world;
    aux::vector mu = p.getDistributedExpectation();
    aux::matrix sum(mu.size(), mu.size(), 0);

    for(unsigned int i = 0 ; i < p.getSize() ;i++) {
        sum += p.getWeight(i)*outer_prod(p.get(i)-mu, p.get(i)-mu);
    }
    sum = boost::mpi::all_reduce(world, sum, std::plus<matrix>());
    return sum/p.getDistributedTotalWeight();
}

aux::matrix calcStdDev(indii::ml::aux::DiracMixturePdf& p)
{
    static int count = 0;
    namespace aux = indii::ml::aux;
    namespace ublas = boost::numeric::ublas;
    namespace lapack = boost::numeric::bindings::lapack;
    boost::mpi::communicator world;
    aux::matrix cov = p.getDistributedCovariance();
    
    aux::vector diag_v(cov.size1());
    int err =  lapack::syev('V', 'U', cov, diag_v);
    if(err != 0) {
        throw(-1);
    }
    aux::matrix tmp;
    
    for(unsigned int i = 0 ; i < diag_v.size() ; i++) {
        if(diag_v[i] < 0) {
            count++;
            if(abs(diag_v[i]) < 1e-10)
                diag_v[i] = 0;
            else
                throw(-5);
        }
        diag_v[i] = sqrt(diag_v[i]);
    }
    ublas::diagonal_matrix<double, ublas::column_major, ublas::unbounded_array<double> >
                diag_m(diag_v.size(), diag_v.data());
    
    tmp = prod(cov, diag_m);
    cov = prod(tmp, trans(cov));

    fprintf(stdout, "(%i)\n", count);
    return cov;
}

/* Run - runs the particle filter
 * pass - variable to pass to callback function
**/
int BoldPF::run(void* pass = NULL)
{
    boost::mpi::communicator world;
    unsigned int rank = world.rank();
//    unsigned int size = world.size();
    using std::endl;

    if(status == ERROR || status == RUNNING || status == DONE) {
        return status;
    }

    *debug << "mu size: " << filter->getFilteredState().getDistributedExpectation().size();
    *debug << "dimensions: " << filter->getFilteredState().getDimensions();
    
    /* Simulation Section */
    double conttime = disctime_s*dt_s;
    unsigned int stim_index = 0;

    *debug << "Starting at " << conttime
                << " disctime_s: " << disctime_s
                << " disctime_l: " << disctime_l 
                << " dt_l: " << dt_l << " dt_s: " << dt_s 
                << " measure size: " << measure.size()
                << " stim size: " << stim.size() << "\n";
    
    if(call_points.start) 
        callback(this, pass);
    
    /* 
     * Run the particle filter either until we reach a predetermined end
     * time, or until we are done processing measurements.
     */
     status = RUNNING;
     while(status == RUNNING && disctime_s*dt_s < dt_l*measure.size()) {
        /* time */
        conttime = disctime_s*dt_s;
//        *debug << "."; 
        
        /* Update Input if there is any*/
        while(stim_index < stim.size() && stim[stim_index].time <= conttime) {
            model->setinput(aux::vector(1, stim[stim_index].level));
            stim_index++;
 //           *debug << conttime;
        }
        

        /* Check to see if it is time to update */
        if(conttime >= disctime_l*dt_l) { 
            //acquire the latest measurement
            *debug << "Measuring at " <<  conttime << "\n";
            aux::vector meas(measure[disctime_l]);
            outputVector(*debug, meas);
            *debug << "\n";

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
                *debug << "\n" << "Error! ESS was " << ess << "\n";
                status = ERROR;
                break;
            } 
            
            //time to resample
            if(ess < ESS_THRESH) {
                *debug << " ESS: " << ess << ", Stratified Resampling\n";

                filter->getFilteredState().distributedNormalise();
                aux::vector tmpmu = filter->getFilteredState().getDistributedExpectation();
                outputVector(*debug, tmpmu);
                *debug << "\n\n";
                
                aux::matrix statecov = filter->getFilteredState().getDistributedCovariance();
                outputMatrix(*debug, statecov);
                *debug << "\n\n";

                aux::matrix stddev;
                try {
                    stddev = calcStdDev(filter->getFilteredState());
                } catch(...) {
                    if(rank == 0){
                        *debug << "Ess: " << ess << endl;
                        *debug << "Mu: ";
                        outputVector(*debug, tmpmu);
                        *debug << endl << "Cov: ";
                        outputMatrix(*debug, statecov);
                        *debug << endl;
                    }
                    status = ERROR;
                    break;
                }
                
                filter->resample(&resampler);
                
                *debug << " ESS: " << ess << ", Regularized Resampling\n\n";
                filter->setFilteredState( resampler_reg->resample(
                                filter->getFilteredState(), stddev) );
            } else {
                *debug << " ESS: " << ess << ", No Resampling Necessary!\n";
            }

            /* Perform updates to delta variables */
            if(method == DELTA) 
                latchBold();
            
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
        status = DONE;
        if(call_points.end) 
            callback(this, pass);
    }

    *debug << "Stop time: "<< conttime << endl;
    
    return status;
};

bool isclose(double a, double b) 
{
    return fabs(a*1000 - b*1000) < 1;
}

void BoldPF::latchBold()
{
    for(unsigned int ii = 0 ; ii < filter->getFilteredState().getSize(); ii++) {
        aux::vector& p = filter->getFilteredState().get(ii);
        aux::vector m = model->measure(p);
        for(unsigned int jj = 0; jj < model->getMeasurementSize(); jj++)
            p[model->getStateSize() - model->getMeasurementSize() + jj] = m[jj];
    }

    //make dirty
    filter->getFilteredState().distributedNormalise();
};

aux::vector bold_mean(const std::vector<aux::vector>& in)
{
    aux::vector sum(in.back().size(), 0);
    for(size_t i = 0 ; i < in.size() ; i++) 
        sum = sum + in[i];
    return sum/in.size();
}

aux::vector bold_stddev(const std::vector<aux::vector>& in, const aux::vector& mean)
{
    aux::vector sum(in.back().size(), 0);
    for(size_t i = 0 ; i < in.size() ; i++)
        sum = sum + scalar_pow(mean - in[i], 2);
    return element_sqrt(sum/in.size());
}

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
            double longstep, std::ostream* output, unsigned int numparticles,
            double shortstep, unsigned int method_p, int weightf, bool flatten) : 
            dt_l(longstep), dt_s(shortstep), 
            disctime_l(0), disctime_s(0), status(UNSTARTED), method(method_p),
            measure(measurements), stim(activations), 
            ESS_THRESH(50)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    
    using std::endl;
    /* initialize debug/output */
    debug = output;

    /* Initalize the model and filter*/
    aux::vector drift;
    aux::vector boldmu = bold_mean(measurements);
    aux::vector boldstd = bold_stddev(measurements, boldmu);
    if(method_p == DC) {
        drift = (measurements[0] + measurements[1] + measurements[2])/3.;
    } else
        drift = aux::vector(measurements.front().size(), 0);

    aux::vector weightv(measurements.front().size(), weightvar);
    model = new BoldModel(weightv, weightf, measurements.front().size(), drift);
    model->setinput(aux::vector(1, 0));
    aux::DiracMixturePdf tmp(model->getStateSize());
    filter = new indii::ml::filter::ParticleFilter<double>(model, tmp);

    /* 
     * Particles Setup 
     */
    *debug << "Generating prior" << std::endl;
    unsigned int localparticles = numparticles/size;
    
    //give excess particles to last rank
    if(rank == (size-1))
        localparticles += numparticles - localparticles*size;

    /* Generate Prior */
    aux::vector width(model->getStateSize());
    for(unsigned int ii = 0 ; ii < model->getMeasurementSize(); ii++) {
        width(model->indexof(model->TAU_S  ,ii)) = 2*.25;
        width(model->indexof(model->TAU_F  ,ii)) = 2*.25;
        width(model->indexof(model->EPSILON,ii)) = .5;
        width(model->indexof(model->TAU_0  ,ii)) = 2*.25;
        width(model->indexof(model->ALPHA  ,ii)) = 2*.045;
        width(model->indexof(model->E_0    ,ii)) = 2*.1;
        width(model->indexof(model->V_0    ,ii)) = 2*.1e-20*.1e-20;

        //Assume they start at 0
        width(model->indexof(model->V_T,ii)) = 1e-20;
        width(model->indexof(model->Q_T,ii)) = 1e-20;
        width(model->indexof(model->S_T,ii)) = 1e-20;
        width(model->indexof(model->F_T,ii)) = 1e-20;
    }

    for(unsigned int ii = model->getStateSize()-model->getMeasurementSize() ;
                    ii < model->getStateSize(); ii++) {
        if(method == DC)
            width[ii] = boldstd[ii-model->getStateSize()+model->getMeasurementSize()]/.4;
        else
            width[ii] = 0;
    }
    model->generatePrior(filter->getFilteredState(), localparticles, width, flatten); 
    filter->getFilteredState().distributedNormalise();
    
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
            
    if(method == DELTA) 
        latchBold();
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

  #ifndef NDEBUG
  unsigned int initialSize = input.getDistributedSize();
  #endif 
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
  
  #ifndef NDEBUG
  unsigned int endSize = input.getDistributedSize();
  #endif 
  aux::vector endMu = input.getDistributedExpectation();
  aux::matrix endCov = input.getDistributedCovariance();
  
  assert (initialSize == endSize);
};

#endif //BOLDPF_H
