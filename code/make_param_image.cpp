#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkMetaDataObject.h"

#include "segment.h"
#include "tools.h"

#include <itkMultiplyImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkRandomImageSource.h>
#include <itkExtractImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include "modNiftiImageIO.h"

#include <sstream>
#include <iostream>
#include <vector>

#include <vcl_list.h>
#include <vul/vul_arg.h>

using namespace std;

Label4DType::Pointer createRegions(double sigma, double threshp,
            Image4DType::Pointer templ = NULL)
{
    itk::RandomImageSource<Image4DType>::Pointer rImage = 
                itk::RandomImageSource<Image4DType>::New();
    rImage->SetNumberOfThreads(10);
    rImage->SetMax(1);
    rImage->SetMin(0);
    itk::BinaryThresholdImageFilter<Image4DType, Image4DType>::Pointer thresh = 
                itk::BinaryThresholdImageFilter<Image4DType, Image4DType>::New();

    unsigned long out_size[4];
    if(templ) {
        out_size[0] = templ->GetRequestedRegion().GetSize()[0];
        out_size[1] = templ->GetRequestedRegion().GetSize()[1];
        out_size[2] = templ->GetRequestedRegion().GetSize()[2];
        out_size[3] = 1;
    } else {
        out_size[0] = 100;
        out_size[1] = 100;
        out_size[2] = 100;
        out_size[3] = 1; 
        
    }
    //outimage
    rImage->SetSize(out_size);
    
    /* Create a Random Field, Perform Smoothing to the set number of resels
     * then threshold, to create the regions
     */
    itk::DiscreteGaussianImageFilter<Image4DType, Image4DType>::Pointer gaussF = 
                itk::DiscreteGaussianImageFilter<Image4DType, Image4DType>::New();
    double variance[3] = {sigma,sigma,sigma};
    gaussF->SetVariance(variance);
    gaussF->SetInput(rImage->GetOutput());
    
    thresh->SetInput(gaussF->GetOutput());
    thresh->SetLowerThreshold(0);
    thresh->SetUpperThreshold(threshp);
    thresh->SetInsideValue(0);
    thresh->SetOutsideValue(1);
    

    //segment image into discrete blobs
    itk::ConnectedComponentImageFilter<Image4DType, Label4DType>::Pointer conn = 
                itk::ConnectedComponentImageFilter<Image4DType, Label4DType>::New();
    conn->SetInput(thresh->GetOutput());

    conn->Update();
    
    if(templ) {
        conn->GetOutput()->SetDirection(templ->GetDirection());
        conn->GetOutput()->SetSpacing(templ->GetSpacing());
    }
    return conn->GetOutput();
}

vector< vector<double> > read_params(string filename) 
{
    FILE* fin = fopen(filename.c_str(), "r");
    
    vector< vector<double> > params;
    if(!fin) 
        return params;

    char* input = NULL;
    size_t size = 0;
    char* prev = NULL;
    char* curr = NULL;
    double parsed = 0;
    while(getline(&input, &size, fin) && !feof(fin)) {
        vector<double> tmp;
        prev = input;
        parsed = strtod(input, &curr);
        while(prev != curr) {
            tmp.push_back(parsed);
            prev = curr;
            parsed = strtod(prev, &curr);
        }

        if(tmp.size() > 0)
            params.push_back(tmp);
        free(input);
        input = NULL;
    }
    fclose(fin);
    
    fprintf(stderr, "Parameter Sets:\n");
    for(size_t i = 0 ; i < params.size(); i++) {
        fprintf(stderr, "%zu: ", i);
        for(size_t j = 0 ; j < params[i].size(); j++) {
            fprintf(stderr, "%f ", params[i][j]);
        }
        fprintf(stderr, "\n");
    }
    return params;
}

Image4DType::Pointer applyParams(string param_f, Label4DType::Pointer regions)
{
    Image4DType::Pointer out = Image4DType::New();
    Image4DType::SizeType size = regions->GetRequestedRegion().GetSize();
    
    /*Determine Size*/
    size[3] = PSIZE;
    out->SetRegions(size);
    out->Allocate();
    out->SetSpacing(regions->GetSpacing());
    out->SetDirection(regions->GetDirection());

    //open and read param_f
    vector< vector<double> > params = read_params(param_f);
    
    //Write each set of parameters to the parameter image
    for(size_t xx = 0 ; xx<regions->GetRequestedRegion().GetSize()[0] ; xx++) {
        for(size_t yy = 0 ; yy<regions->GetRequestedRegion().GetSize()[1] ; yy++) {
            for(size_t zz = 0 ; zz<regions->GetRequestedRegion().GetSize()[2] ; zz++) {
                Image4DType::IndexType index = {{xx, yy, zz, 0}};
                int i = (int)regions->GetPixel(index);
                if(i == 0) { 
                    writeVector<double, std::vector<double> >(out, 3, 
                                params[0], index);
                } else {
                    writeVector<double, std::vector<double> >(out, 3, 
                                params[i%(params.size()-1)+1], index);
                }
            }
        }
    }

    return out;
}

//The labelmap should already have been masked through a maxprob image for
//graymatter
//TODO: Make the first element in each time series the section label
int main( int argc, char **argv ) 
{
    /* Input related */
    vul_arg<string> a_templ("-t" ,"Template (takes orientation/spacing/brain)");
    vul_arg<string> a_imageOut(0 ,"Image Out");
    vul_arg<double> a_thresh("-T" ,"Threshold for region gen.", .53);
    vul_arg<double> a_sigma("-s" ,"Sigma for gaussian filter, region gen", 4);
    vul_arg<string> a_params(0 ,"File with parameters, 1 region per line, nonactive on first line"
                "<TAU_0> <ALPHA> <E_0> <V_0> <TAU_S> <TAU_F> <EPSILON> <A_1> <A_2>");
    vul_arg<string> a_noise("-n" ,"File with per-param variance (if file not supplied"
                " no noise will be added, file should be single line:"
                "<TAU_0> <ALPHA> <E_0> <V_0> <TAU_S> <TAU_F> <EPSILON> <A_1> <A_2>");
   
    /* Processing */
    vul_arg_parse(argc, argv);
    
    Image4DType::Pointer templ = NULL;
    
    /* Template Image */
    if(a_templ() != "") {
        itk::ImageFileReader<Image4DType>::Pointer reader = 
                    itk::ImageFileReader<Image4DType>::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        reader ->SetFileName( a_templ() );
        templ = reader->GetOutput();
        templ->Update();
    }
    
    fprintf(stderr, "Creating Regions\n");
    Label4DType::Pointer regions = createRegions(a_sigma(), a_thresh(), templ);
    printf("\nSpacing %f %f %f\n", regions->GetSpacing()[0],regions->GetSpacing()[1], 
                regions->GetSpacing()[2]);
    fprintf(stderr, "Done\n");
    
    {
    fprintf(stderr, "Outputing regions\n");
    itk::ImageFileWriter<Label4DType>::Pointer writer = 
                itk::ImageFileWriter<Label4DType>::New();
    writer->SetInput(regions);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName("regions.nii.gz");
    writer->Update();
    }
    fprintf(stderr, "Done\n");

    fprintf(stderr, "Creating Param Image\n");
    Image4DType::Pointer out = applyParams(a_params(), regions);
    fprintf(stderr, "Done\n");
    
    fprintf(stderr, "Outputing Parameters\n");
    {
    itk::ImageFileWriter<Image4DType>::Pointer writer = 
                itk::ImageFileWriter<Image4DType>::New();
    writer->SetInput(out);
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(a_imageOut());
    writer->Update();
    fprintf(stderr, "Done\n");
    }

    return 0;
}

