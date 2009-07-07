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

/* Typedefs */
typedef double ImagePixelType;
typedef itk::Image< ImagePixelType,  4 > Image4DType;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

#define TIMEDIM 3
#define SERIESDIM 0

//write a vector to a dimension of an image
void writeVector(Image4DType::Pointer out, int dir, const aux::vector& input, 
            Image4DType::IndexType start)
{
    itk::ImageLinearIteratorWithIndex<Image4DType> 
                it(out, out->GetRequestedRegion());
    it.SetDirection(dir);
    it.SetIndex(start);

    size_t i;
    for(i = 0 ; i < input.size() && !it.IsAtEndOfLine() ; i++) {
        it.Set(input[i]);
        ++it;
    }

    assert(i==input.size() && it.IsAtEndOfLine());
}

//write particles out to a dimesion of an image
void writeParticles(Image4DType::Pointer out, 
            const std::vector<aux::DiracPdf>& elems, int t)
{
    Image4DType::IndexType pos = {{0, 0, 0, t}};
    
    for(size_t i = 0 ; i<elems.size() ; i++) {
        pos[2] = i;
        writeVector(out, 1, elems[i].getExpectation(), pos);
    }
}

//dir1 should be the first matrix dimension, dir2 the second
void writeMatrix(Image4DType::Pointer out, int dir1, int dir2, 
            const aux::matrix& input, Image4DType::IndexType start)
{
    itk::ImageSliceIteratorWithIndex< Image4DType > 
                it(out, out->GetRequestedRegion());
    it.SetFirstDirection(dir1);
    it.SetSecondDirection(dir2);
    
    it.SetIndex(start);

    for(size_t j = 0 ; j < input.size2() && !it.IsAtEndOfSlice() ; j++) {
        for(size_t i = 0 ; i < input.size1() && !it.IsAtEndOfLine() ; i++) {
            it.Set(input(i,j));
        }
        it.NextLine();
    }
}

//read dimension of image into a vector
void readVector(const Image4DType::Pointer in, int dir, aux::vector& input, 
            Image4DType::IndexType start)
{
    itk::ImageLinearConstIteratorWithIndex<Image4DType> 
                it(in, in->GetRequestedRegion());
    it.SetDirection(dir);
    it.SetIndex(start);

    size_t i;
    for(i = 0 ; i < input.size() && !it.IsAtEndOfLine() ; i++) {
        input[i] = it.Get();
    }

    assert(i==input.size());
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

    vul_arg<unsigned> a_num_particles("-p", "Number of particles.", 30000);
    vul_arg< vcl_vector<string> > a_seriesfiles("-tsv", "2D timeseries files (one"
                " per section");
    vul_arg< string > a_seriesfile("-ts", "2D timeseries file", "");
    vul_arg<unsigned> a_divider("-div", "Intermediate Steps between samples.", 64);
    vul_arg<string> a_stimfile("-stim", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<string> a_serialofile("-so", "Where to put a serial output file", "");
    vul_arg<string> a_serialifile("-si", "Where to find a serial input file", "");
    vul_arg<bool> a_expweight("-expweight", "Use exponential weighting function", false);
    vul_arg<bool> a_avgweight("-avgweight", "Average weights rather than multiply", false);
    vul_arg<double> a_resampness("-r", "Ratio of total particles that the ESS must "
                               "reach for the filter to resample. Ex .8 Ex2. .34", .5);
    vul_arg<string> a_boldfile("-bo", "Where to put bold image file", "bold.nii.gz");
    vul_arg<string> a_statefile("-so", "Where to put state image file","state.nii.gz");
    vul_arg<string> a_covfile("-co", "Where to put covariance image file", "");

    vul_arg_parse(argc, argv);
    if(rank == 0) {
        out = &cout;
    } else {
        out = &nullout;
    }

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);

    std::ifstream fin;

    double sampletime = 2;
    
    ImageReaderType::Pointer reader;
    itk::ImageLinearIteratorWithIndex<Image4DType> iter;
    Image4DType::Pointer measInput;

    if(rank == 0) {
        
        /* Open up the input */
        reader = ImageReaderType::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        cout << a_stimfile() << endl;;
        reader->SetFileName( a_stimfile() );
        reader->Update();
        measInput = reader->GetOutput();
        
        /* Create the iterator, to move forward in time for a particlular section */
        iter = itk::ImageLinearIteratorWithIndex<Image4DType>(measInput, 
                    measInput->GetRequestedRegion());
        iter.SetDirection(3);
        Image4DType::IndexType index = {{0, 0, 0, 0}};
        iter.SetIndex(index);
        
        /* Create a model */
        string str;
        itk::ExposeMetaData(measInput->GetMetaDataDictionary(), 
                    "TemporalResolution", sampletime);
        cout << sampletime << endl;;
        if(sampletime == 0) {
            cout << "Image Had Invalid Temporal Resolution!" << endl;
            cout << "Using 2 seconds!" << endl;
            sampletime=2;
        }
        cout << left << setw(20) << "TR" << ": " << sampletime << endl;
    
        /* Generate Prior */
        if(!a_serialifile().empty()) {
            std::ifstream serialin(a_serialifile().c_str(), std::ios::binary);
            boost::archive::binary_iarchive inArchive(serialin);
            inArchive >> tmpX;
            if(a_num_particles() != tmpX.getSize())
                cout << "Number of particles changed to " << tmpX.getSize();
            a_num_particles() = tmpX.getSize();
//        } else if(cheating) {
//            model.generatePrior(tmpX, a_num_particles(), cheat);
        } else {
            *out << "Generating prior" << endl;
            model.generatePrior(tmpX, a_num_particles());
//            aux::matrix tmp = tmpX.getDistributedCovariance();
//            *out << "Covariance: " << endl;
//            outputMatrix(*out, tmp);
//            *out << endl;
        }
    
    }
        
    boost::mpi::broadcast(world, tmpX, 0);

    /* Divide Initial Distribution among nodes */
    for(size_t i = rank ; i < a_num_particles() ; i+= size) {
        x0.add(tmpX.get(i));
    }

    /* Create the filter */
    indii::ml::filter::ParticleFilter<double> filter(&model, x0);
  
    /* create resamplers */
    /* Normal resampler, used to eliminate particles */
    indii::ml::filter::StratifiedParticleResampler resampler(a_num_particles());

    /* Regularized Resample */
    aux::Almost2Norm norm;
    aux::AlmostGaussianKernel kernel(model.getStateSize(), 1);
    RegularizedParticleResamplerMod< aux::Almost2Norm, 
                aux::AlmostGaussianKernel > resampler_reg(norm, kernel, &model);


    Image4DType::Pointer measOutput, stateOutput, covOutput, partOutput;

    int meassize = 0;

    /////////////////////////////////////////////////////////////////////
    // output setup 
    /////////////////////////////////////////////////////////////////////
    if(rank == 0) {
        //BOLD
        measOutput = Image4DType::New();
        init4DImage(measOutput , measInput->GetRequestedRegion().GetSize()[0],
                1, 1, measInput->GetRequestedRegion().GetSize()[3]);
        measOutput->SetMetaDataDictionary(measInput->GetMetaDataDictionary());

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
        meassize = measInput->GetRequestedRegion().GetSize()[0];
    }

    boost::mpi::broadcast(world, sampletime, 0);
    boost::mpi::broadcast(world, meassize, 0);
    
    BoldModel model(a_expweight(), a_avgweight());

    aux::DiracMixturePdf x0(model.getStateSize());
    
    /* Full Distribution */
    aux::DiracMixturePdf tmpX(model.getStateSize());


    /* Simulation Section */
    aux::DiracMixturePdf distr(model.getStateSize());
    aux::vector input(1);
    aux::vector meas(meassize);
    aux::vector mu(model.getStateSize());
    aux::symmetric_matrix cov(model.getStateSize());
    input[0] = 0;
    double nextinput;
    int disctime = 0;
    bool done = false;
    int tmp = 0;
    if(rank == 0 && !a_stimfile().empty()) {
        fin.open(a_stimfile().c_str());
        fin >> nextinput;
    }

    while(!done) {
#ifdef PARTOUT
        if( rank == 0 && !partfile.empty() ) {
            const std::vector<aux::DiracPdf>& particles = 
                        filter.getFilteredState().getAll();
            writeParticles(partOutput, particles, disctime);
        }
#endif //PARTOUT
        
        //TODO maybe this should be split up to prevent ranks from having
        //to move in lock-step
        if(rank == 0 && !fin.eof() && disctime*sampletime/a_divider() >= nextinput) {
            fin >> input[0];
            fin >> nextinput;
        }

        boost::mpi::broadcast(world, input, 0);
        model.setinput(input);

        /* time for update */
        *out << "t= " << disctime*sampletime/a_divider() << ", ";
        
        if(disctime%a_divider() == 0) { //time for update!
            //acquire the latest measurement
            if(rank == 0) {
                cout << "Measuring at " <<  disctime/a_divider() << endl;
                Image4DType::IndexType index = {{0, 0, 0, disctime/a_divider()}};
                readVector(measInput, 0, meas, index);
                outputVector(std::cerr, meas);
                cerr << endl;
                ++iter;
                done = iter.IsAtEndOfLine();
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
                cerr << "Total Weight: " << filter.getFilteredState().getTotalWeight() << endl;
                aux::vector weights = filter.getFilteredState().getWeights();
                outputVector(cerr, weights);
                exit(-5);
//            } else {
//                cerr << "Total Weight: " << filter.getFilteredState().getTotalWeight() << endl;
//                aux::vector weights = filter.getFilteredState().getWeights();
//                outputVector(cerr, weights);
            }

            //time to resample
            if(ess < a_num_particles()*a_resampness()) {
                *out << endl << " ESS: " << ess << ", Deterministic Resampling" << endl;
                filter.resample(&resampler);
                
                *out << " ESS: " << ess << ", Regularized Resampling" << endl << endl;
                filter.resample(&resampler_reg);
            } else
                *out << endl << " ESS: " << ess << ", No Resampling Necessary!" << endl;
        
            /* Get state */
            distr = filter.getFilteredState();
            mu = distr.getDistributedExpectation();
            cov = distr.getDistributedCovariance();
            if( rank == 0 ) {
                /* output measurement */
                
                //save states in an image
                Image4DType::IndexType index = {{0, 0, 0, disctime/a_divider()}};
                writeVector(measOutput, 0, model.measure(mu), index);
                writeVector(stateOutput, 1, mu, index);
                writeMatrix(covOutput, 1, 2, cov, index);
            }

        } else { //no update available, just step update states
            filter.filter(disctime*sampletime/a_divider());
        }
   
       
        /* Update Time, disctime */
        tmp = disctime;
        boost::mpi::broadcast(world, disctime, 0);
        if(tmp != disctime) {
            cerr << "ERROR ranks have gotten out of sync at " << disctime << endl;
            exit(-1);
        }
        disctime++;
    }
    printf("Index at end: %ld %ld \n", iter.GetIndex()[0], iter.GetIndex()[1]);


    x0 = filter.getFilteredState();
    if( rank == 0 ) {
        WriterType::Pointer writer = WriterType::New();
        writer->SetImageIO(itk::modNiftiImageIO::New());
        //serialize
        if(!a_serialofile().empty()) {
            std::ofstream serialout(a_serialofile().c_str(), std::ios::binary);
            boost::archive::binary_oarchive outArchive(serialout);
            outArchive << x0;
        }

        if(!a_boldfile().empty()) {
            writer->SetFileName(a_boldfile());  
            writer->SetInput(measOutput);
            writer->Update();
        }

        if(!a_statefile().empty()) {
            writer->SetFileName(a_statefile());  
            writer->SetInput(stateOutput);
            writer->Update();
        }
        
        if(!a_covfile().empty()) {
            writer->SetFileName(a_covfile());  
            writer->SetInput(covOutput);
            writer->Update();
        }
    }

  return 0;

}

