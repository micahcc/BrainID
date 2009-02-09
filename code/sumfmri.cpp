//image readers
#include "itkOrientedImage.h"
#include "itkImageFileReader.h"

//test
#include "itkImageFileWriter.h"
#include "itkImageConstIteratorWithIndex.h"

//iterators
#include "itkImageLinearConstIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"

//standard libraries
#include <ctime>
#include <cstdio>
#include <list>
#include <sstream>

//reading DICOM
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkMetaDataDictionary.h"
#include "itkNaryElevateImageFilter.h"

// declare images
typedef signed short PixelType;
typedef itk::OrientedImage<PixelType, 2> Image2DType;
typedef itk::OrientedImage<PixelType, 3> Image3DType;
typedef itk::OrientedImage<PixelType, 4> Image4DType;

typedef itk::ImageSliceIteratorWithIndex< Image4DType > SliceIterator4D;
typedef itk::ImageLinearIteratorWithIndex< Image4DType > PixelIterator4D;
typedef itk::ImageLinearIteratorWithIndex< Image3DType > PixelIterator3D;
typedef itk::ImageLinearIteratorWithIndex< Image2DType > PixelIterator2D;

typedef struct {
    int label;
    std::list<SliceIterator4D> list;
} SectionType ;

void test_alltogether(Image4DType::Pointer fmri_img, std::string filename, 
            std::list<SectionType*>& active_voxels); 
void test_times(Image4DType::Pointer fmri_img, std::string filename);
void test_sections(Image4DType::Pointer fmri_img, std::string filename,
            std::list<SectionType*>& active_voxels);


SectionType* findLabel(std::list<SectionType*>& list, int label) 
{
//    fprintf(stderr, "searching for %i\n", label);
    std::list<SectionType*>::iterator it = list.begin();
    while(it != list.end() ) {
        if((*it)->label == label) 
            return *it;
        it++;
    }
    return NULL;
}

//get the location of input_it
//set that location in the reference images
//check to see if the the probability is above a threshold and
//if it is, based on the region in labelmap, add the list
bool checkAddPixel(const Image4DType::Pointer fmri_img, 
            SliceIterator4D fmri_it, std::list< SectionType* >& active_voxels,
            const Image3DType::Pointer labelmap_img) 
{
    PixelIterator3D labelmap_it( labelmap_img, labelmap_img->GetRequestedRegion() );
    
    Image4DType::PointType fmri_phys;
    Image3DType::PointType phys_3D;

    Image3DType::IndexType labelmap_index;

    fmri_img->TransformIndexToPhysicalPoint( fmri_it.GetIndex(), fmri_phys);
    phys_3D[0] = fmri_phys[0];
    phys_3D[1] = fmri_phys[1];
    phys_3D[2] = fmri_phys[2];
    
//    printf("Pixel at: %f %f %f", phys_3D[0], phys_3D[1], phys_3D[2]);

    if(labelmap_img->TransformPhysicalPointToIndex(phys_3D, labelmap_index)) {
        labelmap_it.SetIndex(labelmap_index);
        if( labelmap_it.Get() == 0) return false;
//        fprintf(stderr, "%li %li %li -> %li %li %li : %li\n", fmri_it.GetIndex()[0],
//                    fmri_it.GetIndex()[1], fmri_it.GetIndex()[2], labelmap_index[0],
//                    labelmap_index[1], labelmap_index[2], labelmap_it.Get());

        SectionType* section;
        if(!( section = findLabel(active_voxels, labelmap_it.Get()) )) {
            fprintf(stderr, "NOT FOUND, appending\n");
            //append to list of active pixels
            active_voxels.push_front(new SectionType);
            section = active_voxels.front();
            section->label = labelmap_it.Get();
        }
        //append to the returned labeltype
        section->list.push_front(fmri_it);
    }
    return true;
}


//Reads a dicom directory then returns a pointer to the image
Image4DType::Pointer read_dicom(std::string directory)
{
    // create elevation filter
    itk::NaryElevateImageFilter<Image3DType, Image4DType>::Pointer elevateFilter
                = itk::NaryElevateImageFilter<Image3DType, Image4DType>::New();

    
    // create name generator and attach to reader
    itk::GDCMSeriesFileNames::Pointer nameGenerator = itk::GDCMSeriesFileNames::New();
    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0020|0100"); // acquisition number
    nameGenerator->SetDirectory(directory);

    // get series IDs
    const std::vector<std::string>& seriesUID = nameGenerator->GetSeriesUIDs();

    // create reader array
    itk::ImageSeriesReader<Image3DType>::Pointer *reader = 
                new itk::ImageSeriesReader<Image3DType>::Pointer[seriesUID.size()];

    // declare series iterators
    std::vector<std::string>::const_iterator seriesItr=seriesUID.begin();
    std::vector<std::string>::const_iterator seriesEnd=seriesUID.end();
    
    //reorder the input based on temporal number.
    while (seriesItr!=seriesEnd)
    {
        itk::ImageSeriesReader<Image3DType>::Pointer tmp_reader = 
                    itk::ImageSeriesReader<Image3DType>::New();

        itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
        tmp_reader->SetImageIO(dicomIO);

        std::vector<std::string> fileNames;

//        printf("Accessing %s\n", seriesItr->c_str());
        fileNames = nameGenerator->GetFileNames(seriesItr->c_str());
        tmp_reader->SetFileNames(fileNames);

        tmp_reader->ReleaseDataFlagOn();

        tmp_reader->Update();
        tmp_reader->GetOutput()->SetMetaDataDictionary(dicomIO->GetMetaDataDictionary());
        
        std::string value;
        dicomIO->GetValueFromTag("0020|0100", value);
        printf("Temporal Number: %s\n", value.c_str());
        
        reader[atoi(value.c_str())-1] = tmp_reader;

        ++seriesItr;
    }
        
    // connect each series to elevation filter
    for(unsigned int ii=0 ; ii<seriesUID.size() ; ii++) {
        elevateFilter->SetInput(ii,reader[ii]->GetOutput());
    }

    //now elevateFilter can be used just like any other reader
    elevateFilter->Update();
    
    //create image readers
    Image4DType::Pointer fmri_img = elevateFilter->GetOutput();
    fmri_img->Update();
    return fmri_img;
}

//The labelmap should already have been masked through a maxprob image for
//graymatter
int main( int argc, char **argv ) 
{
    // check arguments
    if(argc != 3) {
        printf("Usage: %s <4D fmri dir> <labels>", argv[0]);
        return EXIT_FAILURE;
    }
    
    Image4DType::Pointer fmri_img = read_dicom(argv[1]);

    //label index
    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
                itk::ImageFileReader<Image3DType>::New();
    labelmap_read->SetFileName( argv[2] );
    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
    labelmap_img->Update();

    std::list< SectionType* > active_voxels;

    //allocate the iterators for the greymater
    //Image3DType::RegionType region_lm = labelmap_img->GetRequestedRegion();
  
    SliceIterator4D fmri_it( fmri_img, fmri_img->GetRequestedRegion() );
    PixelIterator4D time_it( fmri_img, fmri_img->GetRequestedRegion() );
  
    fmri_it.SetFirstDirection(0);
    fmri_it.SetSecondDirection(1);
    fmri_it.SetIndex(fmri_img->GetRequestedRegion().GetIndex());
    
    time_it.SetDirection(3);
    time_it.SetIndex(fmri_img->GetRequestedRegion().GetIndex());
    ++time_it;

    while(fmri_it != time_it) {
        while(!fmri_it.IsAtEndOfSlice()) {
            while(!fmri_it.IsAtEndOfLine()) {
                checkAddPixel(fmri_img, fmri_it, active_voxels, labelmap_img);
                ++fmri_it;
            }
            fmri_it.NextLine();
        }
        fmri_it.NextSlice();
    }

    //now to sum everything up...
    test_alltogether(fmri_img, "all_", active_voxels);
    test_times(fmri_img, "times_");
    test_sections(fmri_img, "sections_", active_voxels);
    
    return 0;
}


////TESTING/////
////////////////////////////////////////////////////
//Testing by writing out everything summed together
void test_alltogether(Image4DType::Pointer fmri_img, std::string filename, 
            std::list<SectionType*>& active_voxels) 
{       
    fprintf(stderr, "showing all active pixels\n");
    Image4DType::RegionType fmri_region = fmri_img->GetRequestedRegion();

    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();

    Image3DType::Pointer outputImage = Image3DType::New();

    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;
    out_size[0] = fmri_region.GetSize()[0];
    out_size[1] = fmri_region.GetSize()[1];
    out_size[2] = fmri_region.GetSize()[2];

    out_index[0] = fmri_region.GetIndex()[0];
    out_index[1] = fmri_region.GetIndex()[1];
    out_index[2] = fmri_region.GetIndex()[2];

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageSliceIteratorWithIndex<Image3DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(1);

    std::list< SectionType* >::iterator section_it = active_voxels.begin();
    std::list<SliceIterator4D>::iterator voxel_it;
    out_it.GoToBegin();

    //Zero out the output image
    while(!out_it.IsAtEnd()) {
        fprintf(stdout, ".");
        while(!out_it.IsAtEndOfSlice()) {
            while(!out_it.IsAtEndOfLine()) {
                out_it.Value() = 0;
                ++out_it;
            }
            out_it.NextLine();
        }
        out_it.NextSlice();
    }

    while(section_it != active_voxels.end()) {
        voxel_it = (*section_it)->list.begin();
        fprintf(stdout, "Label: %u Number: %u\n", (*section_it)->label, 
                (*section_it)->list.size());
        while(voxel_it != (*section_it)->list.end()) {
            out_index[0] = voxel_it->GetIndex()[0];
            out_index[1] = voxel_it->GetIndex()[1];
            out_index[2] = voxel_it->GetIndex()[2];
//            fprintf(stderr, "%li %li %li\n", out_index[0], out_index[1], out_index[2]);
            out_it.SetIndex(out_index);
            out_it.Value() = voxel_it->Get();
            voxel_it++;
        }


        section_it++;
    }

    writer->SetFileName( filename + ".nii.gz" );  
    writer->SetInput(outputImage);
    writer->Update();
}

////////////////////////////////////////////////////
//Testing by writing out each timestep as a 3D image
void test_times(Image4DType::Pointer fmri_img, std::string filename)
{
    fprintf(stderr, "Showing every time in fmri image\n");
    Image4DType::RegionType fmri_region = fmri_img->GetRequestedRegion();

    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();

    Image3DType::Pointer outputImage = Image3DType::New();

    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;
    out_size[0] = fmri_region.GetSize()[0];
    out_size[1] = fmri_region.GetSize()[1];
    out_size[2] = fmri_region.GetSize()[2];

    out_index[0] = fmri_region.GetIndex()[0];
    out_index[1] = fmri_region.GetIndex()[1];
    out_index[2] = fmri_region.GetIndex()[2];

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
//    outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageSliceIteratorWithIndex<Image3DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(1);

    std::ostringstream os;
    
    SliceIterator4D fmri_it( fmri_img, fmri_img->GetRequestedRegion() );
  
    fmri_it.SetFirstDirection(0);
    fmri_it.SetSecondDirection(1);
    fmri_it.GoToBegin();
    
    PixelIterator4D time_it( fmri_img, fmri_img->GetRequestedRegion() );
    time_it.SetDirection(3);
    time_it.GoToBegin();
    ++time_it;
    int count = 0;
    do {
        //zero output
        out_it.GoToBegin();
        while(!out_it.IsAtEnd()) {
            fprintf(stderr, ".");
            while(!out_it.IsAtEndOfSlice()) {
                while(!out_it.IsAtEndOfLine()) {
                    out_it.Value() = 0;
                    ++out_it;
                }
                out_it.NextLine();
            }
            out_it.NextSlice();
        }

        while(fmri_it != time_it && !fmri_it.IsAtEnd()) {
            while(!fmri_it.IsAtEndOfSlice()) {
                while(!fmri_it.IsAtEndOfLine()) {
                    out_it.Value() = fmri_it.Get();
                    ++fmri_it;
                    ++out_it;
                }
                fmri_it.NextLine();
                out_it.NextLine();
            }
            fmri_it.NextSlice();
            out_it.NextSlice();
        }

        os.str("");
        os << filename << count++ << ".nii.gz";
        fprintf(stderr, "%s\n", os.str().c_str());
        writer->SetFileName( os.str() );  
        writer->SetInput(outputImage);
        writer->Update();
        ++time_it;
    } while(!time_it.IsAtEnd() && !fmri_it.IsAtEnd());

//    fprintf(stderr, "%li %li %li %li and %li %li %li %li\n", fmri_it.GetIndex()[0],
//            fmri_it.GetIndex()[1], fmri_it.GetIndex()[2], fmri_it.GetIndex()[3], 
//            time_it.GetIndex()[0], time_it.GetIndex()[1], time_it.GetIndex()[2],
//            time_it.GetIndex()[3]);
}

///////////////////////////////////////////////////
//testing code, write out an image for each section
void test_sections(Image4DType::Pointer fmri_img, std::string filename,
            std::list<SectionType*>& active_voxels) 
{
    fprintf(stderr, "Writing out every section, from active voxel list\n");
    // writer
    Image4DType::RegionType fmri_region = fmri_img->GetRequestedRegion();

    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();

    Image3DType::Pointer outputImage = Image3DType::New();

    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;
    out_size[0] = fmri_region.GetSize()[0];
    out_size[1] = fmri_region.GetSize()[1];
    out_size[2] = fmri_region.GetSize()[2];

    out_index[0] = fmri_region.GetIndex()[0];
    out_index[1] = fmri_region.GetIndex()[1];
    out_index[2] = fmri_region.GetIndex()[2];

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageSliceIteratorWithIndex<Image3DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(1);

    std::ostringstream os;
    std::list< SectionType* >::iterator section_it = active_voxels.begin();
    std::list<SliceIterator4D>::iterator voxel_it;
    while(section_it != active_voxels.end()) {
        out_it.GoToBegin();
        while(!out_it.IsAtEnd()) {
            fprintf(stderr, ".");
            while(!out_it.IsAtEndOfSlice()) {
                while(!out_it.IsAtEndOfLine()) {
                    out_it.Value() = 0;
                    ++out_it;
                }
                out_it.NextLine();
            }
            out_it.NextSlice();
        }

        voxel_it = (*section_it)->list.begin();
        fprintf(stdout, "Label: %u Number: %u\n", (*section_it)->label, 
                (*section_it)->list.size());
        while(voxel_it != (*section_it)->list.end()) {
            out_index[0] = voxel_it->GetIndex()[0];
            out_index[1] = voxel_it->GetIndex()[1];
            out_index[2] = voxel_it->GetIndex()[2];
            fprintf(stdout, "%li %li %li\n", out_index[0],
                    out_index[1], out_index[2]);
            out_it.SetIndex(out_index);
            out_it.Value() = voxel_it->Get();
            voxel_it++;
        }

        os.str("");
        os << filename << (*section_it)->label << ".nii.gz";
        fprintf(stderr, "writing: %s\n", os.str().c_str());
        writer->SetFileName( os.str() );  
        writer->SetInput(outputImage);
        writer->Update();

        section_it++;
    }
}
