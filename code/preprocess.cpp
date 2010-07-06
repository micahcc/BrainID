#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkDivideImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>

#include "tools.h"
#include "segment.h"
#include "modNiftiImageIO.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>

#include <vcl_list.h>
#include <vul/vul_arg.h>

typedef itk::AddImageFilter< Image4DType, Image4DType, Image4DType > AddF4;
typedef itk::DivideImageFilter< Image4DType, Image4DType, Image4DType > DivF4;
typedef itk::SubtractImageFilter< Image4DType > SubF4;
typedef itk::ImageFileReader< Image4DType >  ImageReaderType;
typedef itk::ImageFileWriter< Image4DType >  WriterType;

using namespace std;

void writeImage(Image4DType::Pointer image, std::string filename)
{
    
    try {
        itk::ImageFileWriter<Image4DType>::Pointer out = 
            itk::ImageFileWriter<Image4DType>::New();
        out->SetInput(image);
        out->SetFileName(filename);
        cout << "Writing: " << filename << endl;
        out->Update();
    } catch(itk::ExceptionObject) {
        cerr << "Error opening " << filename << endl;
        exit(-4);
    }
}

Image4DType::Pointer preprocess_help(Image4DType::Pointer input, 
            std::vector<Activation>& stim, double sampletime, unsigned int erase,
            bool nospline, bool smart, std::string base = "")
{
//    boost::mpi::communicator world;
//    const unsigned int rank = world.rank();
    /* Set up measurements image */
    //*output << "Conditioning FMRI Image" << endl;
    //remove first 2 time step, since they are typically polluted
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
        input = deSplineBlind(input, input->GetRequestedRegion().GetSize()[3]/15, base);
        input = dc_bump(input);
    }
    std::cerr << "Done." << endl;

    return input;
}

/* Main Function */
int main(int argc, char* argv[])
{
    vul_arg<string> a_input(0, "4D timeseries file");
    vul_arg<string> a_output(0, "output base (may be a directory)");
    
    vul_arg<bool> a_smart("-S", "Use \"smart\" knots based on less active regions"
                , false);
    vul_arg<bool> a_delta("-l", "Use deltas between measurements, this precludes"
                "the drift option", false);
    
    vul_arg<string> a_stimfile("-s", "file containing \"<time> <value>\""
                "pairs which give the time at which input changed", "");
    vul_arg<double> a_timestep("-t", "TR (timesteps in 4th dimension)", 2);
    vul_arg<double> a_erase("-E", "Erase this many volumes", 2);
    
    vul_arg_parse(argc, argv);
    
    vul_arg_display_usage("No Warning, just echoing");

    ///////////////////////////////////////////////////////////////////////////////
    //Done Parsing, starting main part of code
    ///////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "Brainid Version: %s\n", BRAINID_VERSION);

    Image4DType::Pointer inImage;
    std::vector<Activation> input;

    Image3DType::Pointer rms;

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
    cout << "Creating Output Images" << endl;
    inImage = preprocess_help(inImage, input, a_timestep(), a_erase(), a_delta(),
                a_smart(), a_output());
    std::string finalout = a_output();
    finalout.append("final.nii.gz");
    writeImage(inImage, finalout);
    return 0;
}

