#include <iostream>
#include <fstream>
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/filter/StratifiedParticleResampler.hpp"
#include "gsl/gsl_randist.h"

namespace aux = indii::ml::aux;
using namespace std;

const unsigned int DIMS = 2;
const double MU[DIMS] = {1, 1};
const double SIGMA[DIMS] = {3, 7};
const unsigned int COMPS = 10000;

int main(int argc, char** argv)
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    ostream* out = &cerr;
    ofstream outfile("/dev/null");
    if(rank != 0 ) {
        out = &outfile; 
    }

    if(rank == 0) {
        double tmp;
    //    cin >>  tmp;
    }
        
    
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_ranlxd2);
    {
        unsigned int seed;
        FILE* file = fopen("/dev/urandom", "r");
        fread(&seed, 1, sizeof(unsigned int), file);
        fclose(file);
        gsl_rng_set(rng, seed^rank);
        *out << "Seeding with " << (unsigned int)(seed^rank) << "\n";
    }

    /* Create an imperical gaussian pdf, 2D */
    aux::DiracMixturePdf newPdf(DIMS);
    aux::vector comp(DIMS);
    for(unsigned int i = 0 ; i < COMPS ; i++) {
        double weight = 1;
        for(unsigned int j = 0 ; j < DIMS ; j++) {
            comp[j] = gsl_ran_flat(rng, -20, 20) + MU[j];
            weight *= gsl_ran_gaussian_pdf(comp[j]-MU[j], SIGMA[j]);
        }
        newPdf.add(comp, weight);
    }
    *out << "Single Node Pdf Size: " << newPdf.getSize() << endl;
    *out << "Overall Pdf Size: " << newPdf.getDistributedSize() << endl;

    aux::symmetric_matrix origvar =  newPdf.getDistributedCovariance();
    aux::vector origmu = newPdf.getDistributedExpectation();
    newPdf.distributedNormalise();
    for(unsigned int i = 0 ; i < origmu.size() ; i++) 
        *out << origmu[i] << " ";
    *out << endl << endl;

    for(unsigned int i = 0 ; i < origvar.size1() ; i++)  {
        for(unsigned int j = 0 ; j < origvar.size2() ; j++) { 
            *out << origvar(i,j) << " ";
        }
        *out << endl;
    }
    *out << endl << newPdf.calculateDistributedEss() << endl;
    *out << endl << endl;

    *out << "Deterministic" << endl;
    indii::ml::filter::StratifiedParticleResampler resamp;
    resamp.useDeterministic();
    aux::DiracMixturePdf resPdf = resamp.resample(newPdf);
    *out << "Finished Resampling" << endl;
    *out << "Single Node Pdf Size: " << resPdf.getSize() << endl;
    *out << "Overall Pdf Size: " << resPdf.getDistributedSize() << endl;

    aux::symmetric_matrix finalvar =  resPdf.getDistributedCovariance();
    *out << "Getting Covariance" << endl;
    aux::vector finalmu = resPdf.getDistributedExpectation();
    for(unsigned int i = 0 ; i < finalmu.size() ; i++) 
        *out << finalmu[i] << " ";
    *out << endl << endl;

    for(unsigned int i = 0 ; i < finalvar.size1() ; i++)  {
        for(unsigned int j = 0 ; j < finalvar.size2() ; j++) { 
            *out << finalvar(i,j) << " ";
        }
        *out << endl;
    }
    *out << endl << resPdf.calculateDistributedEss() << endl;
    *out << endl << endl;
    
    *out << "Pull from custom" << endl;
    resamp.useCustom();
    resPdf = resamp.resample(newPdf);
    *out << "Finished Resampling" << endl;
    *out << "Single Node Pdf Size: " << resPdf.getSize() << endl;
    *out << "Overall Pdf Size: " << resPdf.getDistributedSize() << endl;

    finalvar =  resPdf.getDistributedCovariance();
    *out << "Getting Covariance" << endl;
    finalmu = resPdf.getDistributedExpectation();
    for(unsigned int i = 0 ; i < finalmu.size() ; i++) 
        *out << finalmu[i] << " ";
    *out << endl << endl;

    for(unsigned int i = 0 ; i < finalvar.size1() ; i++)  {
        for(unsigned int j = 0 ; j < finalvar.size2() ; j++) { 
            *out << finalvar(i,j) << " ";
        }
        *out << endl;
    }
    *out << endl << resPdf.calculateDistributedEss() << endl;
    *out << endl << endl;
}
