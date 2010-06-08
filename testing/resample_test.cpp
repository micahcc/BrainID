#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "indii/ml/aux/DiracMixturePdf.hpp"
#include "indii/ml/filter/StratifiedParticleResampler.hpp"
#include "gsl/gsl_randist.h"

namespace aux = indii::ml::aux;
using namespace std;

const unsigned int DIMS = 2;
const unsigned int BINSX = 20;
const unsigned int BINSY = 20;
const double MU[DIMS] = {1, 1};
const double SIGMA[DIMS] = {3, 7};
const unsigned int COMPS = 500;

aux::DiracMixturePdf generate()
{
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    /* Generate an empirical PDF */
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_ranlxd2);
    {
        unsigned int seed;
        FILE* file = fopen("/dev/urandom", "r");
        fread(&seed, 1, sizeof(unsigned int), file);
        fclose(file);
        gsl_rng_set(rng, seed^rank);
        cout << "Seeding with " << (unsigned int)(seed^rank) << "\n";
    }

    /* Create an imperical gaussian pdf, 2D */
    aux::DiracMixturePdf newPdf(DIMS);
    aux::vector comp(DIMS);
    for(unsigned int i = 0 ; i < COMPS ; i++) {
        double weight = 1;
        for(unsigned int j = 0 ; j < DIMS ; j++) {
            comp[j] = gsl_ran_flat(rng, 0, 20);
            weight *= gsl_ran_gamma_pdf(comp[j], 2, 4);
        }
        newPdf.add(comp, weight);
    }
    return newPdf;
}

void makehist(const aux::DiracMixturePdf& in, double hist[BINSX][BINSY], 
            std::string filename)
{
    //find max/min
    vector<double> min(DIMS);
    vector<double> max(DIMS);
    min[0] = in.get(0)[0];
    min[1] = in.get(0)[1];
    max[0] = in.get(0)[0];
    max[1] = in.get(0)[1];
    for(unsigned int j = 0 ; j < DIMS ; j++) {
        for(unsigned int i = 0 ; i < in.getSize(); i++) {
            if(!(min[j] < in.get(i)[j]))
                min[j] = in.get(i)[j];
            if(!(max[j] > in.get(i)[j]))
                max[j] = in.get(i)[j];
        }
    }
    
    for(unsigned int i = 0 ; i < DIMS; i++) {
        min[i] += .000001;
        max[i] += .000001;
    }
    

    for(unsigned int i = 0 ; i < in.getSize(); i++) {
        hist[(int)(BINSX*(in.get(i)[0]-min[0])/(max[0] - min[0]))]
                    [(int)(BINSY*(in.get(i)[1]-min[1])/(max[1] - min[1]))] += in.getWeight(i);
    }

    ofstream fout(filename.c_str(), ios_base::trunc);
    fout << setw(20) << min[0] << setw(20) << max[0]; 
    fout << setw(20) << min[1] << setw(20) << max[1] << endl;
    for(unsigned int i = 0 ; i < BINSX ; i++) {
        for(unsigned int j = 0 ; j < BINSY ; j++) {
            fout << setw(20) << hist[i][j];
        }
        fout << endl;
    }
    fout.close();
}

void clearhist(double toclear[BINSX][BINSY])
{
    for(unsigned int i = 0 ; i < BINSX ; i++) {
        for(unsigned int j = 0 ; j < BINSX ; j++) {
            toclear[i][j] = 0;
        }
    }
}

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

    indii::ml::filter::StratifiedParticleResampler resamp;
    double hist[BINSX][BINSY];
        
    aux::DiracMixturePdf pdf = generate();
    pdf.gatherToNode(0);
    if(rank == 0) {
        clearhist(hist);
        makehist(pdf, hist, "original");
    }
    pdf.redistributeBySize();
    
    resamp.useDeterministic();
    aux::DiracMixturePdf detPdf = resamp.resample(pdf);
    detPdf.gatherToNode(0);
    if(rank == 0) {
        clearhist(hist);
        makehist(detPdf, hist, "default");
    }
    detPdf.redistributeBySize();

    resamp.useCustom();
    aux::DiracMixturePdf myPdf = resamp.resample(pdf);
    myPdf.gatherToNode(0);
    if(rank == 0) {
        clearhist(hist);
        makehist(myPdf, hist, "mine");
    }
    myPdf.redistributeBySize();
}
