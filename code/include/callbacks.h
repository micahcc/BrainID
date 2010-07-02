#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <itkOrientedImage.h>
#include "BoldPF.h"
#include <float.h>

#include "boost/mpi/operations.hpp"

/*******************************************************
 * Definitions/Function Declarations
 *******************************************************/
//base
struct cb_data
{
    unsigned int pos[3];
};

//callbacks to save measurements
struct cb_meas_data
{
    unsigned int pos[3];
    itk::OrientedImage<float, 4>::Pointer measmu;
    itk::OrientedImage<float, 4>::Pointer measvar;
};

int cb_meas_call(BoldPF* bold, void* data);
void cb_meas_init(cb_meas_data* cdata, BoldPF::CallPoints* cp,
            itk::OrientedImage<float, 4>::SizeType size);

//callback to save ALL particles
//0 - param
//1 - particle
//2 - not used
//3 - time
struct cb_part_data
{
    unsigned int pos[3];
    itk::OrientedImage<float, 4>::Pointer image;
    itk::OrientedImage<float, 4>::SizeType size;
    itk::OrientedImage<float, 4>::IndexType prev;
    std::string output;
};

void cb_part_init(cb_part_data* cdata, BoldPF::CallPoints* cp,
            int parameters, int particles, int time, std::string outbase);
int cb_part_call(BoldPF* bold, void* data);

//callbacks to save mu/var for measurements and parameters
struct cb_all_data
{
    unsigned int pos[3];
    itk::OrientedImage<float, 4>::Pointer measmu;
    itk::OrientedImage<float, 4>::Pointer measvar;
    itk::OrientedImage<float, 5>::Pointer parammu;
    itk::OrientedImage<float, 6>::Pointer paramvar;
};

void cb_all_init(cb_all_data* cdata, BoldPF::CallPoints* cp,
            Image4DType::SizeType size, int parameters);
int cb_all_call(BoldPF* bold, void* data);

//callbacks to save histogram
//dimensions 0-3 what you expect, with 1 extra time value for the initial
//dimension 4 sets parameter | measurement
//dimension 5 sets histogram element [concentration0 conc1 conc2 ... start stop mu]
struct cb_hist_data
{
    unsigned int pos[3];
    unsigned int size;
    itk::OrientedImage<float, 6>::Pointer histogram;
};

void cb_hist_init(cb_all_data* cdata, BoldPF::CallPoints* cp,
            Image4DType::SizeType size, int parameters, int histcount);
int cb_hist_call(BoldPF* bold, void* data);

/********************************************************
 * Implementation of the above functions
 * 
 * needs to go to a cpp file eventually
*********************************************************/
#include "callbacks.h"
#include <itkImageFileWriter.h>
#include <sstream>
#include <string>

#include <boost/mpi/operations.hpp>

//measurement saving callback
int cb_meas_call(BoldPF* bold, void* data)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    
    cb_meas_data* cdata = (struct cb_meas_data*)data;

    itk::OrientedImage<float, 4>::IndexType index;
    for(unsigned int i = 0 ; i < 3 ; i++)
        index[i] = cdata->pos[i];
    index[3] = bold->getDiscTimeL();
    
    DiracMixturePdf& dist = bold->getDistribution();
    const aux::vector& weights = bold->getDistribution().getWeights();
    /* Mean */
    aux::vector measmu(bold->getModel().getMeasurementSize(),0); 
    for(unsigned int jj = 0 ; jj < dist.getSize() ; jj++) {
        if(weights[jj] > 0)
            measmu += dist.getWeight(jj)*bold->getModel().measure(dist.get(jj));
    }
    
    /* Variance */
    aux::vector measvar(measmu.size(), 0);
    aux::vector delta(measmu.size(), 0);
    for(unsigned int i = 0 ; i < weights.size(); i++) {
        if(weights[i] > 0) {
            delta = (bold->getDistribution().get(i) - measmu);
            const aux::matrix cov = outer_prod(delta, delta);
            measvar += weights[i]*diag(cov);
        }
    }
    measvar = boost::mpi::all_reduce(world, measvar, std::plus<aux::vector>())
                /bold->getDistribution().getDistributedTotalWeight();

    if(rank == 0) {
         cdata->measmu->SetPixel(index, measmu[0]);
         cdata->measvar->SetPixel(index, measvar[0]);
//         outputVector(std::cout, mu);
//         std::cout << "\n" << meas[0] << "\n";
    }
    return 0;
}

void cb_meas_init(cb_meas_data* cdata, BoldPF::CallPoints* cp,
            itk::OrientedImage<float, 4>::SizeType size)
{
    itk::OrientedImage<float, 4>::Pointer measmu = itk::OrientedImage<float, 4>::New();
    measmu->SetRegions(size);
    measmu->Allocate();
    measmu->FillBuffer(0);
    cdata->measmu = measmu;
    
    itk::OrientedImage<float, 4>::Pointer measvar = itk::OrientedImage<float, 4>::New();
    measvar->SetRegions(size);
    measvar->Allocate();
    measvar->FillBuffer(0);
    cdata->measvar = measvar;
    
    cp->start = false;
    cp->postMeas = true;
    cp->postFilter = false;
    cp->end = false;
}

//particle saving callback
void cb_part_init(cb_part_data* cdata, BoldPF::CallPoints* cp,
            int parameters, int particles, int time, std::string outbase)
{
    cdata->size[0] = parameters+1;
    cdata->size[1] = particles;
    cdata->size[2] = 1;
    cdata->size[3] = time;
    cdata->output = outbase;

    for(unsigned int i = 0 ; i < 4 ; i++)
        cdata->prev[i] = 0;
    
    cdata->image = itk::OrientedImage<float, 4>::New();
    cdata->image->SetRegions(cdata->size);
    cdata->image->Allocate();
       
    //set callback points
    cp->start = false;
    cp->postMeas = true;
    cp->postFilter = false;
    cp->end = false;
}

int cb_part_call(BoldPF* bold, void* data)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    struct cb_part_data* cdata = (struct cb_part_data*)data;
   
    //check to see if this is a different xyz position, if so
    //write out the old image
    itk::OrientedImage<float, 4>::IndexType index;
    for(unsigned int i=0; i < 3; i++)
        index[i] = cdata->pos[i];
    index[3] = 0;
    if(index != cdata->prev) {
        std::ostringstream oss(cdata->output);
        for(int i = 0 ; i < 3 ; i++)
            oss << cdata->prev[i] << "_";
        oss << ".nii";
        itk::ImageFileWriter<itk::OrientedImage<float, 4> >::Pointer writer = 
                    itk::ImageFileWriter<itk::OrientedImage<float, 4> >::New();
        writer->SetFileName(oss.str());
        writer->SetInput(cdata->image);
        writer->Update();
        cdata->prev = index;
    }
    
    index[0] = 0; //param
    index[1] = 0; //particle
    index[2] = 0;
    index[3] = bold->getDiscTimeL();
  
    std::vector< std::vector< DiracPdf > > xsFull;
    std::vector< aux::vector > wsFull;

    boost::mpi::gather(world, bold->getDistribution().getAll(), xsFull, 0); 
    boost::mpi::gather(world, bold->getDistribution().getWeights(), wsFull, 0); 

    if(rank == 0) {
        for(size_t rank_ii = 0 ; rank_ii < size ; rank_ii++) { 
            for(size_t ee = 0 ; ee < xsFull[rank_ii].size() ; ee++) {
                index[0] = 0;
                for(size_t mm = 0 ; mm < cdata->size[0] ; mm++) {
                    //index1 - particle, index0 param
                    cdata->image->SetPixel(index, xsFull[rank_ii][ee].
                                getExpectation()[mm]); 
                    index[0]++;
                }
                //write weight after the parameters
                cdata->image->SetPixel(index, wsFull[rank_ii][ee]);
                index[1]++;  //doesn't get reset across particles...
            }
        }
    }
    return 0;
}

//save everything callback
void cb_all_init(cb_all_data* cdata, BoldPF::CallPoints* cp,
            Image4DType::SizeType size, int parameters)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();

    itk::OrientedImage<float, 4>::Pointer measmu, measvar;
    itk::OrientedImage<float, 5>::Pointer parammu;
    itk::OrientedImage<float, 6>::Pointer paramvar;
    
    if(rank == 0) {
        measmu = itk::OrientedImage<float, 4>::New();
        measvar = itk::OrientedImage<float, 4>::New();
        measmu->SetRegions(size);
        measvar->SetRegions(size);

        itk::OrientedImage<float, 5>::SizeType size5 = {{size[0], size[1], size[2], 
                    size[3], parameters}};
        itk::OrientedImage<float, 6>::SizeType size6 = {{size[0], size[1], size[2], 
                    size[3], parameters, parameters}};
        
        parammu = itk::OrientedImage<float, 5>::New();
        parammu->SetRegions(size5);
        
        paramvar = itk::OrientedImage<float, 6>::New();
        paramvar->SetRegions(size6);

        measmu->Allocate();
        measvar->Allocate();
        parammu->Allocate();
        paramvar->Allocate();
        
        measmu->FillBuffer(0);
        measvar->FillBuffer(0);
        parammu->FillBuffer(0);
        paramvar->FillBuffer(0);
    }

    cdata->measmu = measmu;
    cdata->measvar = measvar;
    cdata->parammu = parammu;
    cdata->paramvar = paramvar;
       
    //set callback points
    cp->start = false;
    cp->postMeas = true;
    cp->postFilter = false;
    cp->end = false;
}

int cb_all_call(BoldPF* bold, void* data)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    
    //shortcuts
    DiracMixturePdf& dist = bold->getDistribution();
    cb_all_data* cdata = (struct cb_all_data*)data;
    
    //set up location
    itk::OrientedImage<float, 4>::IndexType index4;
    itk::OrientedImage<float, 5>::IndexType index5;
    itk::OrientedImage<float, 6>::IndexType index6;
    for(unsigned int i = 0 ; i < 3 ; i++) {
        index4[i] = cdata->pos[i];
        index5[i] = cdata->pos[i];
        index6[i] = cdata->pos[i];
    }
    index4[3] = bold->getDiscTimeL();
    index5[3] = bold->getDiscTimeL();
    index6[3] = bold->getDiscTimeL();
    
    const aux::vector& weights = bold->getDistribution().getWeights();
    
    /* Get Expected Values/variances: */
    aux::vector parammu = bold->getDistribution().getDistributedExpectation();
    aux::vector measmu(bold->getModel().getMeasurementSize(),0); 
    for(unsigned int jj = 0 ; jj < dist.getSize() ; jj++) {
        if(weights[jj] > 0)
            measmu += dist.getWeight(jj)*bold->getModel().measure(dist.get(jj));
    }
    measmu = boost::mpi::all_reduce(world, measmu, std::plus<aux::vector>())/
                dist.getDistributedTotalWeight();
    assert(measmu.size() == 1);
    
    /* Get variance of parameters */
    aux::symmetric_matrix paramvar = bold->getDistribution().
                getDistributedCovariance();

    //calculate output variance
    double measvar = 0;
    double diff = 0;
    for(unsigned int i = 0 ; i < weights.size(); i++) {
        if(weights[i] > 0) {
            diff = (bold->getDistribution().get(i) - measmu)[0];
            measvar += weights[i]*diff*diff;
        }
    }
    measvar = boost::mpi::all_reduce(world, measvar, std::plus<double>())
                /bold->getDistribution().getDistributedTotalWeight();

    if(rank == 0) {
         cdata->measmu->SetPixel(index4, measmu[0]);
         cdata->measvar->SetPixel(index4, measvar);
         for(index5[4] = 0 ; index5[4] < (int)parammu.size() ; index5[4]++) {
            cdata->parammu->SetPixel(index5, parammu[index5[4]]);
         }
         for(index6[4] = 0 ; index6[4] < (int)paramvar.size1() ; index6[4]++) {
            for(index6[5] = 0 ; index6[5] < (int)paramvar.size1() ; index6[5]++) {
                cdata->paramvar->SetPixel(index6, paramvar(index6[4], index6[5]));
            }
         }
    }
    return 0;
}

//save histogram callback
void cb_hist_init(cb_hist_data* cdata, BoldPF::CallPoints* cp,
            Image4DType::SizeType size, int parameters, int meassize, int histcount)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();

    itk::OrientedImage<float, 6>::Pointer histogram;
    
    if(rank == 0) {
        //parameters, then measurement, histcount +3 for the min/max of the bars, and mean
        itk::OrientedImage<float, 6>::SizeType size6 = {{size[0], size[1], size[2], 
                    size[3]+1, parameters + meassize, histcount + 3}}; 
        
        
        histogram = itk::OrientedImage<float, 6>::New();
        histogram->SetRegions(size6);

        histogram->Allocate();
        histogram->FillBuffer(0);
    }

    cdata->histogram = histogram;
    cdata->size = histcount;
    
    //set callback points
    cp->start = true;
    cp->postMeas = true;
    cp->postFilter = false;
    cp->end = false;
}

double truemeas(BoldPF* bold, unsigned int p, unsigned int n)
{
//    if(bold->isDC())
//        return bold->getModel().measure(bold->getDistribution().get(p))[n] - 
//                    bold->getDistribution().get(p)[bold->getModel().getStateSize()-
//                    bold->getModel().getMeasurementSize()+n];
//    else
        return bold->getModel().measure(bold->getDistribution().get(p))[n];
}


int cb_hist_call(BoldPF* bold, void* data)
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    
    cb_hist_data* cdata = (struct cb_hist_data*)data;

    itk::OrientedImage<float, 6>::IndexType index;
    for(unsigned int i = 0 ; i < 3 ; i++) {
        index[i] = cdata->pos[i];
    }
    index[3] = bold->getDiscTimeL()+(bold->getStatus()==BoldPF::RUNNING ? 1 : 0);
    index[4] = 0;
    index[5] = 0;

    DiracMixturePdf& dist = bold->getDistribution();
    assert(dist.getSize() > 0);
    
    /* Get Expected Values/variances: */
    aux::vector parammu = bold->getDistribution().getDistributedExpectation();
    
    aux::vector measmu =  dist.getWeight(0)*bold->getModel().measure(dist.get(0));
    for(unsigned int jj = 1 ; jj < dist.getSize() ; jj++) {
        if(dist.getWeight(jj) > 0)
            measmu += dist.getWeight(jj)*bold->getModel().measure(dist.get(jj));
    }
    measmu = boost::mpi::all_reduce(world, measmu, std::plus<aux::vector>())/
                dist.getDistributedTotalWeight();

    //put parameters into bins
    aux::vector bins(cdata->size, 0);
    
    /* Make Histogram for Parameters */
    for(unsigned int j = 0 ; j < dist.getDimensions() ; j++) {
        //calculate min/max 
        double min = dist.get(0)[j];
        double max = dist.get(0)[j];
        for(unsigned int i = 0 ; i < dist.getSize(); i++) {
            if(min > dist.get(i)[j] && dist.getWeight(i) > 0)
                min = dist.get(i)[j];
            if(max < dist.get(i)[j] && dist.getWeight(i) > 0)
                max = dist.get(i)[j];
        }
        min = boost::mpi::all_reduce(world, min, boost::mpi::minimum<double>());
        max = boost::mpi::all_reduce(world, max, boost::mpi::maximum<double>());
        max += 1e-8;
     
        //put each parameter in a bin, then get the total
        double binsize = (max-min)/cdata->size;
        for(unsigned int i = 0 ; i < dist.getSize(); i++) {
            if(dist.getWeight(i) > 0)
                bins[(int)((dist.get(i)[j] - min)/binsize)] += dist.getWeight(i);
        }
        
        for(unsigned int i = 0 ; i < cdata->size ; i++) {
            bins[i] = boost::mpi::all_reduce(world, bins[i], std::plus<double>());
        }

        //write out histogram
        if(rank == 0) {
            index[4] = j;
            for(index[5] = 0 ; index[5] < cdata->size; index[5]++) {
                cdata->histogram->SetPixel(index, bins[index[5]]);
            }
            cdata->histogram->SetPixel(index, min);
            index[5]++;
            cdata->histogram->SetPixel(index, max);
            index[5]++;
            cdata->histogram->SetPixel(index, parammu[j]);
        }
    }
    
    /* Make Histogram for Measurements */
    for(unsigned int j = 0 ; j < bold->getModel().getMeasurementSize() ; j++) {
        //calculate min/max 
        double min = truemeas(bold, 0, j); 
        double max = truemeas(bold, 0, j);
        for(unsigned int i = 0 ; i < dist.getSize(); i++) {
            if(min > truemeas(bold, i, j) && dist.getWeight(i) > 0)
                min = truemeas(bold, i, j);
            if(max < truemeas(bold, i, j) && dist.getWeight(i) > 0)
                max = truemeas(bold, i, j);
        }
        min = boost::mpi::all_reduce(world, min, boost::mpi::minimum<double>());
        max = boost::mpi::all_reduce(world, max, boost::mpi::maximum<double>());
        max += 1e-8; //so that the max fits in the top bin
     
        //put each parameter in a bin, then get the total
        double binsize = (max-min)/cdata->size;
        for(unsigned int i = 0 ; i < dist.getSize(); i++) {
            if(dist.getWeight(i) > 0) {
                bins[(int)((truemeas(bold, i, j) - min)/binsize)] 
                            += dist.getWeight(i);
            }
        }
        
        for(unsigned int i = 0 ; i < cdata->size ; i++) {
            bins[i] = boost::mpi::all_reduce(world, bins[i], std::plus<double>());
        }

        //write out histogram
        if(rank == 0) {
            index[4] = bold->getModel().getStateSize() + j;
            for(index[5] = 0 ; index[5] < cdata->size; index[5]++) {
                cdata->histogram->SetPixel(index, bins[index[5]]);
            }
            cdata->histogram->SetPixel(index, min);
            index[5]++;
            cdata->histogram->SetPixel(index, max);
            index[5]++;
            cdata->histogram->SetPixel(index, measmu[j]);
            
        }
    }

    return 0;
}

#endif //CALLBACKS_H
