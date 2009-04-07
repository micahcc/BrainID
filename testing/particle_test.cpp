#include <itkOrientedImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include <indii/ml/filter/ParticleFilter.hpp>
#include <indii/ml/filter/ParticleFilterModel.hpp>
#include <indii/ml/filter/StratifiedParticleResampler.hpp>
#include <indii/ml/aux/GaussianPdf.hpp>
#include <indii/ml/aux/vector.hpp>
#include <indii/ml/aux/matrix.hpp>

#include "BoldModel.hpp"

#define SYSTEM_SIZE 2
#define MEAS_SIZE 1
#define INPUT_SIZE 1
#define ACTUAL_SIZE 3
#define STEPS 250
#define NUM_PARTICLES 1000

#define A1 3.4
#define A2 1.0

using namespace std;
using namespace indii::ml::filter;

namespace aux = indii::ml::aux;

typedef float ImagePixelType;
typedef itk::Image< ImagePixelType,  2 > ImageType;
typedef itk::ImageFileReader< ImageType >  ImageReaderType;
typedef itk::ImageFileWriter< ImageType >  WriterType;

//States:
//0 - v_t
//1 - q_t
//2 - s_t
//3 - f_t

int main(int argc, char* argv[])
{
}
