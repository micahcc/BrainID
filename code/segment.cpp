//image readers
#include "itkImageFileReader.h"

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

#include "sumfmri.h"

//Just returns the pointer stored in the passed list with the 
//matching label
SectionType* findLabel(std::list<SectionType*>& list, int label) 
{
#ifdef DEBUG
    fprintf(stderr, "searching for %i\n", label);
#endif //DEBUG
    std::list<SectionType*>::iterator it = list.begin();
    while(it != list.end() ) {
        if((*it)->label == label) 
            return *it;
        it++;
    }
    return NULL;
}

//checkAddPixel checks in the
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

#ifdef DEBUG
    printf("Pixel at: %f %f %f", phys_3D[0], phys_3D[1], phys_3D[2]);
#endif //DEBUG
    if(labelmap_img->TransformPhysicalPointToIndex(phys_3D, labelmap_index)) {
        labelmap_it.SetIndex(labelmap_index);
        if( labelmap_it.Get() == 0) return false;
#ifdef DEBUG
        fprintf(stderr, "%li %li %li -> %li %li %li : %li\n", fmri_it.GetIndex()[0],
                    fmri_it.GetIndex()[1], fmri_it.GetIndex()[2], labelmap_index[0],
                    labelmap_index[1], labelmap_index[2], labelmap_it.Get());
#endif //DEBUG

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

//sort_voxels fills the list given with new SectionType structs, each of 
//which represents a label from the labelmap image. It then finds each
//member voxel of each label and fills the list in the SectionType
//with iterators for the member voxels.
void sort_voxels(const Image4DType::Pointer fmri_img, 
            const Image3DType::Pointer label_img,
            std::list< SectionType* >& voxels)
{
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
                checkAddPixel(fmri_img, fmri_it, voxels, label_img);
                ++fmri_it;
            }
            fmri_it.NextLine();
        }
        fmri_it.NextSlice();
    }
}

void free_voxels(std::list< SectionType* >& voxels)
{
    do {
        delete voxels.front();
        voxels.pop_front();
    } while (!voxels.empty());
}

//Reads a dicom directory then returns a pointer to the image
//does some of this memory need to be freed??
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

//example main:
//The labelmap should already have been masked through a maxprob image for
//graymatter
//int main( int argc, char **argv ) 
//{
//    // check arguments
//    if(argc != 3) {
//        printf("Usage: %s <4D fmri dir> <labels>", argv[0]);
//        return EXIT_FAILURE;
//    }
//    
//    Image4DType::Pointer fmri_img = read_dicom(argv[1]);
//
//    //label index
//    itk::ImageFileReader<Image3DType>::Pointer labelmap_read = 
//                itk::ImageFileReader<Image3DType>::New();
//    labelmap_read->SetFileName( argv[2] );
//    Image3DType::Pointer labelmap_img = labelmap_read->GetOutput();
//    labelmap_img->Update();
//
//    std::list< SectionType* > active_voxels;
//
//    sort_voxels(fmri_img, labelmap_img, active_voxels);
//    
//    return 0;
//}

