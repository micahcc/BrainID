#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkMetaDataObject.h>
#include <itkDivideImageFilter.h>
#include <itkSubtractImageFilter.h>

#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "tools.h"
#include "modNiftiImageIO.h"
#include "BoldPF.h"
#include "callbacks.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>

#include <vcl_vector.h>
#include <vul/vul_arg.h>

using namespace std;

const double VARTHRESH = .01;

typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

typedef itk::ImageLinearIteratorWithIndex<Image4DType> ImgIter;
typedef itk::DivideImageFilter< Image4DType, Image4DType, Image4DType > DivF4;
typedef itk::SubtractImageFilter< Image4DType > SubF4;

namespace aux = indii::ml::aux;
typedef indii::ml::filter::ParticleFilter<double> Filter;

/* 
 * checks to see if "point" in mask is greater than 0 in "maskimg"
 * returns true of the given point is something greater than 0
 */
bool checkmask(Label4DType::Pointer maskimg, Image4DType::PointType point)
{
    if(!maskimg) return true;

    Image4DType::IndexType index;
    maskimg->TransformPhysicalPointToIndex(point, index);
    if(maskimg->GetRequestedRegion().IsInside(index) && 
                    maskimg->GetPixel(index) > 0) {
        return true;
    }

    return false;
}

/* Helper function to write an image "out" to prefix + filename */
template <typename ImgType>
void writeImage(typename ImgType::Pointer out, std::string prefix, 
            std::string filename)
{
    prefix.append(filename);
    typename itk::ImageFileWriter<ImgType>::Pointer writer = 
        itk::ImageFileWriter<ImgType>::New();
    writer->SetInput(out);
    writer->SetFileName(prefix);
    cout << "Writing: " << prefix<< endl;
    writer->Update();
}

/* 
 * output - a std::vector of aux::vector measurements to be filled
 * input  - an image with measurements to pull from starting at pos
 * pos    - the first index to read measurements from, will read the whole line
 * delta  - calculate the delta between the measurements, and save that instead
 */
void fillvector(std::vector< aux::vector >& output, Image4DType* input,
            Image4DType::IndexType pos, bool delta)
{       
    ImgIter iter(input, input->GetRequestedRegion());
    iter.SetDirection(3);
    iter.SetIndex(pos);

    output.resize(input->GetRequestedRegion().GetSize()[3]);
    if(delta) {
        int i = 1;
        double prev = iter.Get();
        ++iter;
        output[0] = aux::vector(1, 0);
        while(!iter.IsAtEndOfLine()) {
            output[i] = aux::vector(1, iter.Get()-prev);
            prev = iter.Get();
            ++iter;
            i++;
        }
    } else {
        int i = 0;
        while(!iter.IsAtEndOfLine()) {
            output[i] = aux::vector(1, iter.Get());
            ++iter;
            i++;
        }
    }
}

/* Helper function to calculate the preprocessed input 
 * input      - FMRI image with raw measurements
 * stim       - stimulus sequenced used by some paths to optimized the spline
 * sampletime - the TR of the FMRI run
 * erase      - the number of initial voxels to remove
 * nospline   - don't calculate a spline, just do %difference
 * smart      - try to optimized knots (not very good)
 * base       - base filename to write output to
*/
Image4DType::Pointer preprocess_help(Image4DType::Pointer input, 
            std::vector<Activation>& stim, double sampletime, unsigned int erase,
            bool nospline, bool smart, std::string base = "")
{
    boost::mpi::communicator world;
    /* Set up measurements image */
    input = pruneFMRI(input, stim, sampletime, erase);

    //calculate %difference, which is used normally for the bold signal
    // or the modified % difference (with spline rather than mean)
    if(nospline) {
        std::cerr << "Changing Image to %difference" << std::endl;
        SubF4::Pointer sub = SubF4::New();   
        DivF4::Pointer div = DivF4::New();
        Image4DType::Pointer mean = extrude(Tmean(input),
                    input->GetRequestedRegion().GetSize()[3]);
        sub->SetInput1(input);
        sub->SetInput2(mean);
        div->SetInput1(sub->GetOutput());
        div->SetInput2(mean);
        div->Update();
        input = div->GetOutput();
    } else if(smart){
        std::cerr << "De-trending, then dividing by mean" << endl;
        input = deSplineByStim(input, stim, sampletime, base);
    } else {
        std::cerr << "De-trending, then dividing by mean" << endl;
        input = deSplineBlind(input, input->GetRequestedRegion().GetSize()[3]/20, base);
        input = dc_bump(input);
    }
    std::cerr << "Done." << endl;

    return input;
}

/* Count the number of valid voxels according to the mask
 * to get an idea of how long the run time will be
 */
int countValid(Image4DType::Pointer fmriimg, Label4DType::Pointer mask)
{
    int count = 0;
    unsigned int xlen = fmriimg->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = fmriimg->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = fmriimg->GetRequestedRegion().GetSize()[2];
    Image3DType::IndexType index3 = {{0, 0, 0}};
    Image4DType::PointType point4;
    Image4DType::IndexType index4 = {{0, 0, 0, 0}};
    /* Calculate parameters for every voxel */
    for(index3[0] = 0 ; index3[0] < xlen ; index3[0]++) {
        for(index3[1] = 0 ; index3[1] < ylen ; index3[1]++) {
            for(index3[2] = 0 ; index3[2] < zlen ; index3[2]++) {
                //initialize some indexes
                for(int i = 0 ; i < 3 ; i++) index4[i] = index3[i];
                index4[3] = 0;
                fmriimg->TransformIndexToPhysicalPoint(index4, point4);
                if(checkmask(mask, point4)) {
                    count++;
                }
            }
        }
    }
    return count;
}

/* Main Function */
int main(int argc, char* argv[])
{
    /* Initialize mpi */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const unsigned int rank = world.rank();
    const unsigned int size = world.size();

    vul_arg<string> a_input(0, "4D timeseries file");
    
    vul_arg<string> a_mask("-m", "3D mask file");
    vul_arg<int> a_dc("-c", "Calculate DC gain as a state variable", false);
    vul_arg<int> a_delta("-l", "Use deltas between measurements, this precludes"
                "the drift option", false);
    vul_arg<int> a_smart("-S", "Use \"smart\" knots based on less active regions"
                , false);
    
    vul_arg<unsigned int> a_num_particles("-p", "Number of particles.", 3000);
    vul_arg<unsigned int> a_divider("-d", "Intermediate Steps between samples.", 128);
    vul_arg<string> a_stimfile("-s", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<int> a_flat("-f", "Flatten prior?", true);
    vul_arg<int> a_weight("-w", "Use weight function: 0:Normal, 1:Laplace, "
                "2:Hyperbolic, 3:Cauchy", 0);
    vul_arg<double> a_scale("-C", "Scale factor wor weight function", 1.);
    vul_arg<double> a_timestep("-t", "TR (timesteps in 4th dimension)", 2);
    vul_arg<string> a_output("-o", "Output prefix", "");
    vul_arg<unsigned int> a_erase("-e", "Number of times to erase at the front", 2);
    vul_arg<int> a_nospline("-N", "No spline?", 0);
    vul_arg< vcl_vector<int> > a_locat("-L", "Perform at a single location, ex -L 3,12,9");
    vul_arg<int> a_callbacktype("-D", "Data to record, 0 - histogram,"
                " 1 - mean/variance of measurements/parameters,"
                " 2 - measurement mean/variance,"
                " 3 - dump all particles, each location will get a different image (untested)", 1);
    
    vul_arg_parse(argc, argv);
    
    if(rank == 0) {
        vul_arg_display_usage("No Warning, just echoing");
    }

    const unsigned int FILTER_PARAMS = 12; //11 normal plus drift
    const unsigned int BASICPARAMS = 7;
    const unsigned int STATICPARAMS = 2;
    const unsigned int RETRIES = 3;
//    const unsigned int ERASE = 2;

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Rank: %u Size: %u\n", rank,size);
    fprintf(stderr, "Brainid Version: %s\n", BRAINID_VERSION);

    Image4DType::Pointer inImage;
    Image4DType::Pointer paramMuImg;
    itk::OrientedImage<float, 5>::Pointer paramVarImg;

    std::vector<Activation> input;

    Image3DType::Pointer rms;
    Label4DType::Pointer mask;
    Image4DType::SizeType outsize;

    ofstream ofile("/dev/null");
    ostream* output;
    if(rank == 0) {
        output = &cout;
    } else {
        output = &ofile;
    }

    /* Open up the input */
    try {
        ImageReaderType::Pointer reader;
        reader = ImageReaderType::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_input() );
        reader->Update();
        inImage = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_input().c_str());
        exit(-1);
    }

    if(!a_mask().empty()) try{
        itk::ImageFileReader<Label4DType>::Pointer reader;
        reader = itk::ImageFileReader<Label4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader->SetFileName( a_mask() );
        reader->Update();
        mask = reader->GetOutput();
    } catch(itk::ExceptionObject) {
        fprintf(stderr, "Error opening %s\n", a_mask().c_str());
        exit(-2);
    }
    
    /* Open Stimulus file */
    if(!a_stimfile().empty()) {
        input = read_activations(a_stimfile().c_str());
        if(input.empty()) 
            return -1;
    }

    if(input.size() == 0) {
        Activation tmp;
        tmp.time = 0;
        tmp.level= 0;
        input = std::vector<Activation>(1,tmp);
    }

    /* Create Output Images */
//    *output << "Creating Output Images" << endl;
//    for(int i = 0 ; i < 3 ; i++)
//        outsize[i] = inImage->GetRequestedRegion().GetSize()[i];
    outsize = inImage->GetRequestedRegion().GetSize();
    outsize[3] = BASICPARAMS + STATICPARAMS;
    
    if(rank == 0) {
        paramMuImg = Image4DType::New();
        paramMuImg->SetRegions(outsize);
        paramMuImg->Allocate();
        paramMuImg->FillBuffer(0);
        paramMuImg->CopyInformation(inImage);
        
        itk::OrientedImage<float, 5>::SizeType size5;
        for(uint32_t i = 0 ; i < 3; i++)
            size5[i] = outsize[i];
        size5[3] = BASICPARAMS;
        size5[4] = BASICPARAMS;

        paramVarImg = itk::OrientedImage<float,5>::New();
        paramVarImg->SetRegions(size5);
        paramVarImg->Allocate();
        copyInformation<Image4DType, itk::OrientedImage<float,5> >(inImage, paramVarImg);
        paramVarImg->FillBuffer(0);
    }
    
    inImage = preprocess_help(inImage, input, a_timestep(), a_erase(), a_delta() || 
                a_nospline(), a_smart(), a_output());
    
    unsigned int xlen = inImage->GetRequestedRegion().GetSize()[0];
    unsigned int ylen = inImage->GetRequestedRegion().GetSize()[1];
    unsigned int zlen = inImage->GetRequestedRegion().GetSize()[2];
    unsigned int tlen = inImage->GetRequestedRegion().GetSize()[3];
    
    Label3DType::Pointer oMask;
    if(rank == 0) try {
    	/* Build Mask showing the results of each Voxel */
        oMask = Label3DType::New();
        Label3DType::SizeType size3 = {{xlen, ylen, zlen}};
        oMask->SetRegions(size3);
        oMask->Allocate();
        oMask->FillBuffer(0);

    	/* Save detrended image */
        itk::ImageFileWriter<Image4DType>::Pointer out = 
                    itk::ImageFileWriter<Image4DType>::New();
        out->SetInput(inImage);
	string outname = a_output();
	outname.append("pfilter_input.nii.gz");
        out->SetFileName(outname);
        cout << "Writing: " << outname << endl;
        out->Update();
    } catch(itk::ExceptionObject) {
        cerr << "Error opening pfilter_input.nii.gz" << endl;
        exit(-4);
    }

    //acquire rms
//    rms = get_rms(inImage);
    
    unsigned int method = BoldPF::DIRECT;
    if(a_delta()) method = BoldPF::DELTA;
//    else if(a_dc()) method = BoldPF::DC;

    //callback variables, to fill in 
    BoldPF::CallPoints callpoints;
    void* cbdata = NULL;
    int (*cbfunc)(BoldPF*, void*) = NULL;
    switch(a_callbacktype()) {
        case 0: {
            *output << "Setting up histogram" << std::endl;
            cb_hist_data* cbd = new cb_hist_data;
            cb_hist_init(cbd, &callpoints, inImage->GetRequestedRegion().GetSize(), 
                        FILTER_PARAMS, 1, 10);
            cbdata = (void*)cbd;
            cbfunc = cb_hist_call;
        } break;
        case 1: {
            *output << "Setting up mean/var images" << std::endl;
            cb_all_data* cbd = new cb_all_data;
            cb_all_init(cbd, &callpoints, inImage->GetRequestedRegion().GetSize(), 
                        FILTER_PARAMS);
            cbdata = (void*)cbd;
            cbfunc = cb_all_call;
        } break;
        case 2: {
            *output << "Setting up mean/var images of meas" << std::endl;
            cb_meas_data* cbd = new cb_meas_data;
            cb_meas_init(cbd, &callpoints, inImage->GetRequestedRegion().GetSize());
            cbdata = (void*)cbd;
            cbfunc = cb_meas_call;
        } break;
        case 3: {
            std::string outname = a_output();
            outname.append("particle_");
            *output << "Setting up mean/var images of meas" << std::endl;
            cb_part_data* cbd = new cb_part_data;
            cb_part_init(cbd, &callpoints, FILTER_PARAMS, a_num_particles(), 
                        inImage->GetRequestedRegion().GetSize()[3], outname);
            cbdata = (void*)cbd;
            cbfunc = cb_part_call;
        } break;
    }
    
    /* Temporary variables used in the loop */
    time_t start = time(NULL);
    Image3DType::IndexType index3 = {{0, 0, 0}};
    Image4DType::PointType point4;
    Image4DType::IndexType index4 = {{0, 0, 0, 0}};
    int result;
    
    aux::vector mu;
    aux::vector var;
    std::vector< aux::vector > meas(tlen, aux::zero_vector(1));

    /* Set constant A1, A2 */
    int total = 0;
    if(a_locat().size() == 3)
        total = 1;
    else 
        total = countValid(inImage, mask);
    int traveled = 0;
    *output << "Total Voxels: " << total << endl;
       
    /* Calculate parameters for every voxel */
    for(index3[0] = 0 ; index3[0] < xlen ; index3[0]++) {
        for(index3[1] = 0 ; index3[1] < ylen ; index3[1]++) {
            for(index3[2] = 0 ; index3[2] < zlen ; index3[2]++) {
                //initialize some indexes
                for(int i = 0 ; i < 3 ; i++) index4[i] = index3[i];
                index4[3] = 0;
                inImage->TransformIndexToPhysicalPoint(index4, point4);
                
                //if requested, only perform test on a single location
                if(a_locat().size() == 3) {
                    if(!(a_locat()[0] == index3[0] && a_locat()[1] == index3[1] 
                        && a_locat()[2] == index3[2]))
                        continue;
                } else if(!checkmask(mask, point4)) 
                    continue;
			
		if(oMask) oMask->SetPixel(index3, 1);

                result = BoldPF::UNSTARTED;

                *output << index3 << endl;
                fillvector(meas, inImage, index4, a_delta());
                aux::vector mu;
                aux::matrix cov;
                
                for(uint32_t tries = 0 ; tries < RETRIES && result != BoldPF::DONE ; tries++) {
                    //create the bold particle filter
                    BoldPF boldpf(meas, input, a_scale(), a_timestep(),
                            output, a_num_particles(), 1./a_divider(), method,
                            a_weight(), a_flat());
                    
                    //set the callback function
                    for(unsigned int j = 0; j < 3 ; j++)
                        ((cb_data*)cbdata)->pos[j] = index4[j];
                    boldpf.setCallBack(callpoints, cbfunc);
                    
                    //run the particle filter
                    result = boldpf.run(cbdata);
                    mu = boldpf.getDistribution().
                                getDistributedExpectation();
                    cov = boldpf.getDistribution().
                                getDistributedCovariance();
                }
                

                if(oMask && paramMuImg && result == BoldPF::DONE) {
                    oMask->SetPixel(index3, 2);
                    
                    itk::OrientedImage<float,5>::IndexType index5;
                    for(uint32_t pp1 = 0 ; pp1 < 3 ; pp1++) 
                        index5[pp1] = index3[pp1];

                    for(uint32_t pp1 = 0 ; pp1 < BASICPARAMS ; pp1++) {
                        index4[3] = index5[3] = pp1;
                        paramMuImg->SetPixel(index4, mu[pp1]);
                        for(uint32_t pp2 = 0 ; pp2 < BASICPARAMS ; pp2++) {
                            index5[4] = pp2;
                            paramVarImg->SetPixel(index5, cov(pp1,pp2));
                        }
                    }
                    
                    aux::vector a12 = BoldModel::getA(mu[BoldModel::E_0]);
                    *output <<"Dumping A" << std::endl;
                    *output << a12[0] << " " << a12[1] << std::endl;
                    index4[3]++;
                    paramMuImg->SetPixel(index4, a12[0]);
                    index4[3]++;
                    paramMuImg->SetPixel(index4, a12[1]);

                }

                //set the output to a standard -1 if BoldPF failed
                //run time calculation
                time_t tmp = time(NULL);
                traveled++;
                *output << "Time Elapsed: " << difftime(tmp, start) << endl
                     << "Time Remaining: " << (total-traveled)*difftime(tmp,start)/
                                (double)traveled 
                     << endl << "Ratio: " << traveled << "/" << total << endl
                     << "Left: " << total-traveled << "/" << total
                     << endl << "Rate: " << difftime(tmp,start)/(double)traveled << endl;
            }
        }
    }
    
    //write final output
    if(rank == 0) {
        copyInformation<Image4DType, Label3DType>(inImage, oMask);
        writeImage<Label3DType>(oMask, a_output(), "statuslabel.nii.gz");
        copyInformation<Image4DType, Image4DType>(inImage, paramMuImg);
        writeImage<Image4DType>(paramMuImg, a_output(), "parammu_f.nii.gz");
        copyInformation<Image4DType, itk::OrientedImage<float,5> >(inImage, paramVarImg);
        writeImage<itk::OrientedImage<float,5> >(paramVarImg, a_output(), "paramvar_f.nii.gz");
        
        switch(a_callbacktype()) {
        case 0: {
            cb_hist_data* cbd = (cb_hist_data*)cbdata;
            
            copyInformation<Image4DType, itk::OrientedImage<DataType, 6> >(
                        inImage, cbd->histogram);
            writeImage<itk::OrientedImage<DataType, 6> >(cbd->histogram, a_output(),
                        "histogram.nii.gz");
        } break;
        case 1: {
            cb_all_data* cbd = (cb_all_data*)cbdata;
            copyInformation<Image4DType, Image4DType>(inImage,cbd->measmu);
            writeImage<itk::OrientedImage<DataType, 4> >(cbd->measmu, a_output(),
                        "measmu.nii.gz");
            copyInformation<Image4DType, Image4DType>(inImage,cbd->measvar);
            writeImage<itk::OrientedImage<DataType, 4> >(cbd->measvar, a_output(),
                        "measvar.nii.gz");
            copyInformation<Image4DType, itk::OrientedImage<DataType, 5> >(inImage,
                        cbd->parammu);
            writeImage<itk::OrientedImage<DataType, 5> >(cbd->parammu, a_output(),
                        "parammu.nii.gz");
            copyInformation<Image4DType, itk::OrientedImage<DataType, 6> >(inImage,
                        cbd->paramvar);
            writeImage<itk::OrientedImage<DataType, 6> >(cbd->paramvar, a_output(),
                        "paramvar.nii.gz");
        } break;
        case 2: {
            cb_meas_data* cbd = (cb_meas_data*)cbdata;
            cbd->measmu->CopyInformation(inImage);
            writeImage<Image4DType>(cbd->measmu, a_output(), "measmu.nii.gz");
            cbd->measvar->CopyInformation(inImage);
            writeImage<Image4DType>(cbd->measvar, a_output(), "measvar.nii.gz");
        } break;
        }
    }

                
    return 0;

}


