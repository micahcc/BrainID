//This code is inspired by/based on Johnston et. al:
//Nonlinear estimation of the Bold Signal
//NeuroImage 40 (2008) p. 504-514
//by Leigh A. Johnston, Eugene Duff, Iven Mareels, and Gary F. Egan

#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkMetaDataObject.h>

#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <indii/ml/aux/Almost2Norm.hpp>
#include <indii/ml/aux/AlmostGaussianKernel.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "tools.h"
#include "modNiftiImageIO.h"
#include "BoldModel.hpp"
#include "RegularizedParticleResamplerMod.hpp"

#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

//namespace opts = boost::program_options;
namespace aux = indii::ml::aux;

typedef double ImagePixelType;
typedef itk::OrientedImage< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;


void gatherToNode(unsigned int dest, aux::DiracMixturePdf& input) {
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
  
  /* post-conditions */
//  if(rank == 0) {
//    cout << "Start Size: " << initialSize << " EndSize: " 
//                << endSize << endl;
//    cout << "Initial Mu: " << endl;
//    outputVector(cout, initialMu);
//    cout << "End Mu: " << endl;
//    outputVector(cout, endMu);;
//    cout << "Local Mu: " << endl;
//    outputVector(cout, input.getExpectation());;
//    
//    cout << "Initial Cov: " << endl;
//    outputMatrix(cout, initialCov);
//    cout << "End Cov: " << endl;
//    outputMatrix(cout, endCov);;
//    cout << "Local Cov: " << endl;
//    outputMatrix(cout, input.getCovariance());;
//    cout << endl;
//  }
  assert (initialSize == endSize);
//  assert (initialMu == endMu);
//  assert (initialCov == endCov);
}


//write particles out to a dimesion of an image
void writeParticles(Image4DType::Pointer out, 
            const std::vector<aux::DiracPdf>& elems, int t)
{
    Image4DType::IndexType pos = {{0, 0, 0, t}};
    
    for(size_t i = 0 ; i<elems.size() ; i++) {
        pos[2] = i;
        writeVector<double>(out, 1, elems[i].getExpectation(), pos);
    }
}

/* init4DImage 
 * sets the ROI for a new image and then calls allocate
 */
void init4DImage(Image4DType::Pointer& out, size_t xlen, size_t ylen, 
            size_t zlen, size_t tlen)
{
    Image4DType::RegionType out_region;
    Image4DType::IndexType out_index;
    Image4DType::SizeType out_size;

    out_size[0] = xlen;
    out_size[1] = ylen;
    out_size[2] = zlen;
    out_size[3] = tlen; 
    
    out_index[0] = 0;
    out_index[1] = 0;
    out_index[2] = 0;
    out_index[3] = 0;
    
    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);
    
    out->SetRegions( out_region );
    out->Allocate();
}


/* Main Function */
int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    ostream* out;
    ofstream nullout("/dev/null");
    ofstream logout;

    vul_arg<unsigned> a_num_particles("-p", "Number of particles.", 30000);
    vul_arg< vcl_vector<string> > a_seriesfiles("-tsv", "2D timeseries files (one"
                " per section");
    vul_arg< string > a_seriesfile("-ts", "2D timeseries file", "");
    vul_arg<unsigned> a_divider("-div", "Intermediate Steps between samples.", 64);
    vul_arg<string> a_stimfile("-stim", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<bool> a_runto("-runtostart", "Run (as opposed to juumping) to start time"
                " if you are loading from serial you will want to skip, if "
                "you removed data from the beginnig of the fmri timeseries, it"
                " MAY be better to run since it could give more state knowledge",
                false);
    vul_arg<double> a_starttime("-tstart", "Initial time", 0);
    vul_arg<double> a_stoptime("-tstop", "Stop time", 0);
    vul_arg<string> a_serialofile("-so", "Where to put a serial output file", "");
    vul_arg<string> a_serialifile("-si", "Where to find a serial input file", "");
    vul_arg<bool> a_expweight("-expweight", "Use exponential weighting function",
                false);
    vul_arg<bool> a_avgweight("-avgweight", "Average weights rather than multiply",
                false);
    vul_arg<double> a_resampratio("-rr", "Ratio of total particles below which ESS "
                "must reach for the filter to resample. Ex .8 Ex2. .34", 0);
    vul_arg<double> a_weightvar("-weightvar", "Variance of weighting function", 
                3.92e-6);
    vul_arg<unsigned int> a_resampnum("-rn", "Absolute ESS below which to resample"
                , 100);
    vul_arg<string> a_boldfile("-yo", "Where to put bold image file", "");
    vul_arg<string> a_statefile("-xo", "Where to put state image file",
                "");
    vul_arg<string> a_covfile("-co", "Where to put covariance image file", "");
    
    vul_arg<string> a_logfile("-log", "Where to log to (default stdout)", "");
    
    vul_arg_parse(argc, argv);
    
    if(rank == 0) {
        if(!a_logfile().empty()) {
            logout.open(a_logfile().c_str());
            if(!logout.is_open()) {
                cerr << "Warning: could not open logfile: " << a_logfile() << endl;
                out = &cout;
            } else {
                out = &logout;
            }
        } else {
            out = &cout;
        }
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

    double sampletime = 2;
    
    ImageReaderType::Pointer reader;
    itk::ImageLinearIteratorWithIndex<Image4DType> iter;
    Image4DType::Pointer measInput, measOutput, stateOutput, covOutput, partOutput;

    int meassize = 0;
    int startlocation = -1;
    int endlocation = -1;

    if(rank == 0) {
        /* Open up the input */
        reader = ImageReaderType::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        *out << a_stimfile() << endl;;
        reader->SetFileName( a_seriesfile() );
        reader->Update();
        measInput = reader->GetOutput();
        meassize = measInput->GetRequestedRegion().GetSize()[SERIESDIM];
        
        /* Create the iterator, to move forward in time for a particlular section */
        iter = itk::ImageLinearIteratorWithIndex<Image4DType>(measInput, 
                    measInput->GetRequestedRegion());
        iter.SetDirection(TIMEDIM);
        Image4DType::IndexType index = {{0, 0, 0, 0}};
        iter.SetIndex(index);
        
        /* Create a model */
        string str;
        itk::ExposeMetaData(measInput->GetMetaDataDictionary(), 
                    "TemporalResolution", sampletime);
        *out << sampletime << endl;;
        if(sampletime == 0) {
            *out << "Image Had Invalid Temporal Resolution!" << endl;
            *out << "Using 2 seconds!" << endl;
            sampletime=2;
        }
        *out << left << setw(20) << "TR" << ": " << sampletime << endl;
    
        //BOLD
        measOutput = Image4DType::New();
        init4DImage(measOutput , meassize,
                1, 1, measInput->GetRequestedRegion().GetSize()[3]);
        measOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());

    }
    
    boost::mpi::broadcast(world, meassize, 0);
    BoldModel model(a_expweight(), a_avgweight(), meassize, a_weightvar());
    
    /* Full Distribution */
    aux::DiracMixturePdf tmpX(model.getStateSize());
    
    if(rank == 0) {
        
        /////////////////////////////////////////////////////////////////////
        // Particles Setup
        /////////////////////////////////////////////////////////////////////
        /* Generate Prior */
        if(!a_serialifile().empty()) {
            *out << "Reading from: " << a_serialifile() << endl;
            std::ifstream serialin(a_serialifile().c_str(), std::ios::binary);
            boost::archive::binary_iarchive inArchive(serialin);
            inArchive >> tmpX;
            if(a_num_particles() != tmpX.getSize()) {
                *out << "Number of particles changed to " << a_num_particles();
                *out << ", redrawing from distribution" << endl;
                aux::DiracMixturePdf tmpX2(model.getStateSize());
                for(size_t i=0 ; i<a_num_particles() ; i++) {
                    tmpX2.add(tmpX.sample(), 1.0);
                }
                tmpX = tmpX2;
            }
//        } else if(cheating) {
//            model.generatePrior(tmpX, a_num_particles(), cheat);
        } else {
            *out << "Generating prior" << endl;
            model.generatePrior(tmpX, a_num_particles(), 9); //3*sigma, squared
//            aux::matrix tmp = tmpX.getDistributedCovariance();
//            *out << "Covariance: " << endl;
//            outputMatrix(*out, tmp);
//            *out << endl;
        }
    

        /////////////////////////////////////////////////////////////////////
        // output setup 
        /////////////////////////////////////////////////////////////////////
        //STATE
        stateOutput = Image4DType::New();
        init4DImage(stateOutput, 1, model.getStateSize(), 1, 
                measInput->GetRequestedRegion().GetSize()[3]);
        stateOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());
        itk::EncapsulateMetaData(stateOutput->GetMetaDataDictionary(),
                    "Dim0", std::string("unused"));
        itk::EncapsulateMetaData(stateOutput->GetMetaDataDictionary(),
                    "Dim1", std::string("systemmean"));
        itk::EncapsulateMetaData(stateOutput->GetMetaDataDictionary(),
                    "Dim2", std::string("unused"));
        itk::EncapsulateMetaData(stateOutput->GetMetaDataDictionary(),
                    "Dim0", std::string("time"));

        //COVARIANCE
        covOutput = Image4DType::New();
        init4DImage(covOutput, 1, 
                model.getStateSize(), model.getStateSize(), 
                measInput->GetRequestedRegion().GetSize()[3]);
        covOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());
        itk::EncapsulateMetaData(covOutput->GetMetaDataDictionary(),
                    "Dim0", std::string("unused"));
        itk::EncapsulateMetaData(covOutput->GetMetaDataDictionary(),
                    "Dim1", std::string("cov"));
        itk::EncapsulateMetaData(covOutput->GetMetaDataDictionary(),
                    "Dim2", std::string("cov"));
        itk::EncapsulateMetaData(covOutput->GetMetaDataDictionary(),
                    "Dim0", std::string("time"));
#ifdef PARTOUT
        //PARTICLES
        partOutput = Image4DType::New();
        init4DImage(partOutput, 1,  model.getStateSize(), a_num_particles(),
                measInput->GetRequestedRegion().GetSize()[3]*a_divider());
        partOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());
        itk::EncapsulateMetaData(partOutput->GetMetaDataDictionary(),
                    "Dim0", std::string("unused"));
        itk::EncapsulateMetaData(partOutput->GetMetaDataDictionary(),
                    "Dim1", std::string("systemval"));
        itk::EncapsulateMetaData(partOutput->GetMetaDataDictionary(),
                    "Dim2", std::string("particle"));
        itk::EncapsulateMetaData(partOutput->GetMetaDataDictionary(),
                    "Dim0", std::string("time"));
#endif //partout
    
    }

    //todo use distribute function from mixturepdf
    boost::mpi::broadcast(world, sampletime, 0);
    boost::mpi::broadcast(world, a_num_particles(), 0);

    tmpX.redistributeBySize();
    
    /* Create the filter */
    indii::ml::filter::ParticleFilter<double> filter(&model, tmpX);

    /* Output the intial parameters */
    aux::vector mu(model.getStateSize());
    aux::symmetric_matrix cov(model.getStateSize());
    mu = filter.getFilteredState().getDistributedExpectation();
    cov = filter.getFilteredState().getDistributedCovariance();
    *out << "Start Mu : " << endl;
    outputVector(*out, mu);
    *out << endl;

    *out << "Start Cov : " << endl;
    outputMatrix(*out, cov);
    *out << endl;
        
  
    /* create resamplers */
    /* Normal resampler, used to eliminate particles */
    indii::ml::filter::StratifiedParticleResampler resampler(a_num_particles());

    /* Regularized Resample */
    aux::Almost2Norm norm;
    aux::AlmostGaussianKernel kernel(model.getStateSize(), 1);
    RegularizedParticleResamplerMod< aux::Almost2Norm, 
                aux::AlmostGaussianKernel > resampler_reg(norm, kernel, &model);

    /* Simulation Section */
    aux::vector input(1);
    aux::vector meas(meassize);
    input[0] = 0;
    double nextinput;
    int disctime = 0;
    double conttime = 0;
    bool done = false;
    int tmp = 0;
    int status = 0;
    if(rank == 0 && !a_stimfile().empty()) {
        fin.open(a_stimfile().c_str());
        if(!fin.is_open()) {
            *out << "Failed to open file " << a_stimfile() << endl;
            return -1;
        }
        fin >> nextinput;
    }

    /* 
     * Fast Forward in time to start time if we are supposed to skip
     * to the first time (otherwise continue to main loop)
     */
    while(!a_runto() && disctime*sampletime/a_divider() < a_starttime()) {
        if(rank == 0 && !fin.eof() && disctime*sampletime/a_divider() 
                    >= nextinput) {
            *out << "FAST FORWARD: t= " << disctime*sampletime/a_divider() << ", " 
                        << endl;
            fin >> input[0];
            fin >> nextinput;
            *out << "New input: " << input[0] << " Next at: " << nextinput << endl;
        }
        /* Inform the filter of the current time, this needs to be the last
         * time that the previous program would have used, so set it before
         * disctime++ */
        filter.setTime(disctime*sampletime/a_divider());
        disctime++;
    }

    /* 
     * Run the particle filter either until we reach a predetermined end
     * time, or until we are done processing measurements.
     */
    while(!done && (a_stoptime() == 0 || disctime*sampletime/a_divider() 
                < a_stoptime()) ) {
#ifdef PARTOUT
        if( rank == 0 && !partfile.empty() ) {
            const std::vector<aux::DiracPdf>& particles = 
                        filter.getFilteredState().getAll();
            writeParticles(partOutput, particles, disctime);
        }
#endif //PARTOUT
        
        /* time */
        conttime = disctime*sampletime/a_divider();
        *out << "t= " << conttime << ", ";
        
        /* Grab New Input if there is any*/
        if(rank == 0 && !fin.eof() && conttime >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
            *out << "New input: " << input[0] << " Next at: " << nextinput << endl;
        }

        boost::mpi::broadcast(world, input, 0);
        model.setinput(input);

        /* Check to see if it is time to update */
        if(disctime%a_divider() == 0 && (!a_runto() || conttime > a_starttime())) { 

            /* Used to cut out all but the relevent parts of the output
             * images at the end. The reason that this is necessary is that,
             * depending on the timestep, the length of the output length may not
             * be predictable without actually looping through. */
            if(startlocation == -1) startlocation = disctime/a_divider();
            endlocation = disctime/a_divider();
            
            //acquire the latest measurement
            if(rank == 0) {
                *out << "Measuring at " <<  disctime/a_divider() << endl;
                Image4DType::IndexType index = {{0, 0, 0, disctime/a_divider()}};
                status = readVector<double>(measInput, 0, meas, index);
                if(status == -1) {
                    *out << "Oops, overshot, this usually happens when start time"
                                << " is greater than the total time. You may want"
                                << " to check the dimensions of the input "
                                << "timeseries." << endl;
                    exit(-7);
                } else if (status == 1) {
                    done = true;
                }
                outputVector(*out, meas);
                *out << endl;
            }

            //send meas and done to other nodes
            boost::mpi::broadcast(world, meas, 0);
            boost::mpi::broadcast(world, done, 0);
            
            //step forward in time, with measurement
            filter.filter(disctime*sampletime/a_divider(), meas);

            //check to see if resampling is necessary
            double ess = filter.getFilteredState().calculateDistributedEss();
            
            //check for errors, could be caused by total collapse of particles,
            //for instance if all the particles go to an unreasonable value like 
            //inf/nan/neg
            if(isnan(ess) || isinf(ess)) {
                *out << "Total Weight: "
                            << filter.getFilteredState().getDistributedTotalWeight()
                            << endl;
                aux::vector weights = filter.getFilteredState().getWeights();
                outputVector(*out, weights);
                exit(-5);
            } else {
                double totalweight = 
                            filter.getFilteredState().getDistributedTotalWeight();
                *out << "Total Weight: " << totalweight << endl;
                if(totalweight >= 1e20 || totalweight <= 1e-20) {
                    filter.getFilteredState().distributedNormalise();
                }
//                aux::vector weights = filter.getFilteredState().getWeights();
//                outputVector(*out, weights);
            }

            //time to resample
            if(ess < a_num_particles()*a_resampratio() || ess < a_resampnum()) {
                cov = filter.getFilteredState().getDistributedCovariance();
                *out << endl << " ESS: " << ess << ", Stratified Resampling" 
                            << endl;
                *out << "Covariance prior to resampling" << endl;
                outputMatrix(*out, cov);
                *out << endl;
                filter.resample(&resampler);
                cov = filter.getFilteredState().getDistributedCovariance();
                *out << "Covariance after to resampling" << endl;
                outputMatrix(*out, cov);
                *out << endl;
                
                *out << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter.resample(&resampler_reg);
            } else {
                *out << endl << " ESS: " << ess << ", No Resampling Necessary!" 
                            << endl;
            }
        
            /* Save output - but only if there is somewhere to ouptput to. Not
             * outputing anything can lead to serious time savings 
             * */

            if( !a_statefile().empty() ) {
                *out << "writing: " << a_statefile() << endl;
                mu = filter.getFilteredState().getDistributedExpectation();
                Image4DType::IndexType index = {{0, 0, 0, disctime/a_divider()}};
                if(rank == 0)
                    writeVector<double>(stateOutput, 1, mu, index);
            }

            if( !a_covfile().empty() ) {
                *out << "writing: " << a_covfile() << endl;
                cov = filter.getFilteredState().getDistributedCovariance();
                Image4DType::IndexType index = {{0, 0, 0, disctime/a_divider()}};
                if(rank == 0)
                    writeMatrix<double>(covOutput, 1, 2, cov, index);
            }

            if( !a_boldfile().empty() ) {
                *out << "writing: " << a_boldfile() << endl;
                Image4DType::IndexType index = {{0, 0, 0, disctime/a_divider()}};
                mu = filter.getFilteredState().getDistributedExpectation();
                if(rank == 0) 
                    writeVector<double>(measOutput, 0, model.measure(mu), index);
            }

        } else { //no update available, just step update states
            filter.filter(disctime*sampletime/a_divider());
        }
   
       
        /* Update Time, disctime */
        tmp = disctime;
        boost::mpi::broadcast(world, disctime, 0);
        if(tmp != disctime) {
            *out << "ERROR ranks have gotten out of sync at " << disctime << endl;
            exit(-1);
        }
        disctime++;
    }

    *out << "Index at end: "<< iter.GetIndex()[0] << "," << iter.GetIndex()[1] << endl;
    *out << "End time: "<< disctime*sampletime/a_divider() << endl;
                
    mu = filter.getFilteredState().getDistributedExpectation();
    cov = filter.getFilteredState().getDistributedCovariance();

    *out << "End Mu parameters: " << endl;
    outputVector(*out, mu);
    *out << endl;
    
    *out << "End Cov parameters: " << endl;
    outputMatrix(*out, cov);
    *out << endl;
           
//    {
//        aux::vector weights = filter.getFilteredState().getWeights();
//        *out << "End Maximum Likelihood: " << endl;
//        double max = 0;
//        int maxloc = 0;
//        for(unsigned int ii=0; ii<weights.size() ; ii++ ){
//            if(weights[ii] > max){
//                max = weights[ii];
//                maxloc = ii;
//            }
//        }
//        outputVector(*out, filter.GetFilteredState(().get(maxloc)));
//        *out << endl;
//    }

//    tmpX = filter.getFilteredState();
    gatherToNode(0, filter.getFilteredState());

    if( rank == 0 ) {
        WriterType::Pointer writer = WriterType::New();
        writer->SetImageIO(itk::modNiftiImageIO::New());
        //serialize
        if(!a_serialofile().empty()) {
            std::ofstream serialout(a_serialofile().c_str(), std::ios::binary);
            boost::archive::binary_oarchive outArchive(serialout);
            outArchive << filter.getFilteredState();

        }

        /* Bold */
        if(!a_boldfile().empty()) {
            writer->SetFileName(a_boldfile());  
            writer->SetInput(prune<double>(measOutput, TIMEDIM, startlocation, 
                        endlocation));
            writer->Update();
        }

        /* State */
        if(!a_statefile().empty()) {
            writer->SetFileName(a_statefile());  
            writer->SetInput(prune<double>(stateOutput, TIMEDIM, startlocation, 
                        endlocation));
            writer->Update();
        }
        
        /* Covariance */
        if(!a_covfile().empty()) {
            writer->SetFileName(a_covfile());  
            writer->SetInput(prune<double>(covOutput, TIMEDIM, startlocation, 
                        endlocation));
            writer->Update();
        }
    }


  return 0;

}

