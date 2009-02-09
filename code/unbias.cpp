/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMRIBiasFieldCorrectionFilterTest.cxx,v $
  Language:  C++
  Date:      $Date: 2007-12-29 13:36:00 $
  Version:   $Revision: 1.13 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include <iostream>
#include "vnl/vnl_vector.h"

#include "itkImageFileReader.h"
#include <itkImageFileWriter.h>
#include "itkMRIBiasFieldCorrectionFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkGaussianImageSource.h"
#include "itkMultivariateLegendrePolynomial.h"
#include "itkCompositeValleyFunction.h"
#include "itkNormalVariateGenerator.h"
#include "itkArray.h"
#include "itkImageFileWriter.h"
#include "itkSphereSpatialFunction.h"

int main(int argc, char* argv[] )
{
  const int Dimension = 3;
  typedef    float ImagePixelType;
  typedef itk::Image< ImagePixelType,  Dimension >   ImageType;
  typedef itk::Image< ImagePixelType,  Dimension >   MaskType;
  typedef itk::ImageFileReader< ImageType >  ImageReaderType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;
  
  ImageReaderType::Pointer reader = ImageReaderType::New();
  reader->SetFileName( argv[1] );
  reader->Update();
 
  // creates a bias correction filter and run it.
  typedef itk::MRIBiasFieldCorrectionFilter<ImageType, ImageType, MaskType> 
    FilterType;

  FilterType::Pointer filter = FilterType::New();
  
//  // creates a bias field
//  typedef itk::MultivariateLegendrePolynomial BiasFieldType;
//  BiasFieldType::DomainSizeType biasSize(3);
//  int biasDegree = 3;
//  ImageType::SizeType imageSize = reader->GetOutput()->GetRequestedRegion().GetSize();
//  biasSize[0] = imageSize[0];
//  biasSize[1] = imageSize[1];
//  biasSize[2] = imageSize[2];
//  BiasFieldType bias(biasSize.size(), 
//                     biasDegree, // bias field degree 
//                     biasSize);
  
//  BiasFieldType::CoefficientArrayType 
//    coefficients(bias.GetNumberOfCoefficients());
//  BiasFieldType::CoefficientArrayType 
//    initCoefficients(bias.GetNumberOfCoefficients());
//  
//  randomGenerator->Initialize( (int) 2003 );
//  for ( unsigned int i = 0; i < bias.GetNumberOfCoefficients(); ++i )
//    {
//    coefficients[i] = ( randomGenerator->GetVariate() + 1 ) * 0.1;
//    initCoefficients[i] = 0;
//    }
//  bias.SetCoefficients(coefficients);

  // To see the debug output for each iteration, uncomment the
  // following line. 
//  filter->DebugOn();
  
  int biasDegree = 3;
  itk::Array<double> classMeans(2);
  itk::Array<double> classSigmas(2);

  classMeans[0] = 10.0;
  classMeans[1] = 200.0;

  classSigmas[0] = 10.0;
  classSigmas[1] = 20.0;
  
  filter->SetInput( reader->GetOutput() );
  filter->IsBiasFieldMultiplicative( false ); // correct with multiplicative bias 
  filter->SetBiasFieldDegree( biasDegree ); // default value = 3
  filter->SetTissueClassStatistics( classMeans, classSigmas );
  //filter->SetOptimizerGrowthFactor( 1.01 ); // default value
  //filter->SetOptimizerInitialRadius( 0.02 ); // default value
  //filter->SetUsingInterSliceIntensityCorrection( true ); // default value
  filter->SetVolumeCorrectionMaximumIteration( 200 ); // default value = 100
  filter->SetInterSliceCorrectionMaximumIteration( 100 ); // default value = 100
  filter->SetUsingSlabIdentification( true ); // default value = false
  filter->SetSlabBackgroundMinimumThreshold( 0 ); // default value
  filter->SetSlabNumberOfSamples( 10 ); // default value 
  filter->SetSlabTolerance(0.0); // default value
  filter->SetSlicingDirection( 2 ); // default value
  filter->SetUsingBiasFieldCorrection( true ); // default value
  filter->SetGeneratingOutput( true ); // default value

//  filter->SetInitialBiasFieldCoefficients(initCoefficients); //default value is all zero

  //timing
  filter->Update();
  
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[2] );  
  writer->SetInput( filter->GetOutput() );
  writer->Update();

  return EXIT_SUCCESS;
}
