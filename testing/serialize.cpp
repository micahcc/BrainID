#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "BoldModel.hpp"
#include "tools.h"

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();
    BoldModel model(indii::ml::aux::zero_vector(1), false);
    indii::ml::aux::vector startmu(model.getStateSize());
    indii::ml::aux::symmetric_matrix startcov(model.getStateSize());
    size_t startsize = 0;
    double starttime = 0;
    
    indii::ml::aux::vector endmu(model.getStateSize());
    indii::ml::aux::symmetric_matrix endcov(model.getStateSize());
    size_t endsize = 0;
    double endtime = 0;
    
    {
        /* Build an example filter with all the normal data 
         * and run for a limited period */
        cout << "Generating prior" << endl;
        aux::DiracMixturePdf tmpX(model.getStateSize());
        model.generatePrior(tmpX, 1000, 9); //3*sigma, squared
        indii::ml::filter::ParticleFilter<double> filter(&model, tmpX);
        for(int i = 0 ; i < 10 ; i ++) {
            filter.filter(i*.0001);
        }
        
        startmu = filter.getFilteredState().getDistributedExpectation();
        startcov = filter.getFilteredState().getDistributedCovariance();
        startsize = filter.getFilteredState().getSize();
        starttime = filter.getTime();
        
        /* Write out the filter to serial file */
        std::ofstream serialout("serial.out", std::ios::binary);
        boost::archive::binary_oarchive outArchive(serialout);
        outArchive << filter;

    }       
    {
        /* Build an example filter with all the normal data 
         * and test to see if the data is properly updated*/
        cout << "Generating prior" << endl;
        aux::DiracMixturePdf tmpX(model.getStateSize());
        indii::ml::filter::ParticleFilter<double> filter(&model, tmpX);
        
        /* Reload from file */
        cout << "Reading from serial.out" << endl;
        std::ifstream serialin("serial.out", std::ios::binary);
        boost::archive::binary_iarchive inArchive(serialin);
        inArchive >> filter;
        /* Print some stats of about the filter to compare with previous 
         */
        endmu = filter.getFilteredState().getDistributedExpectation();
        endcov = filter.getFilteredState().getDistributedCovariance();
        endsize = filter.getFilteredState().getSize();
        endtime = filter.getTime();
    }

    bool pass = true;
    for(size_t i = 0 ; i < endmu.size() ; i++) {
        if(fabs(startmu[i] - endmu[i]) > .001) {
            pass = false;
            break;
        }
    }
    for(size_t i = 0 ; i < endcov.size1() ; i++) {
        for(size_t j = 0 ; j < endcov.size2() ; j++) {
            if(fabs(startcov(i,j) - endcov(i,j)) > .001) {
                pass = false;
                break;
            }
        }
    }

    if(starttime != endtime) pass = false;
    if(startsize != endsize) pass = false;

    if(pass)
        cout << "pass" << endl;
    else
        cout << "fail" << endl;
}
