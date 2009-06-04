//image readers
#include "itkImageFileReader.h"

//standard libraries
#include <ctime>
#include <cstdio>
#include <list>
#include <sstream>

#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkMetaDataDictionary.h>
#include <itkNaryElevateImageFilter.h>

#include "segment.h"

bool compare_lt(SectionType first, SectionType second)
{
    if(first.label < second.label) 
        return true;
    if(first.label > second.label) 
        return false;
    for(int i = 3 ; i >= 0 ; i--) {
        if(first.point.GetIndex()[i] < second.point.GetIndex()[i]) return true;
        if(first.point.GetIndex()[i] > second.point.GetIndex()[i]) return false;
    }
    return false;
}

//sort_voxels fills the list given with new SectionType structs, each of 
//which represents a label from the labelmap image. It then finds each
//member voxel of each label and fills the list in the SectionType
//with iterators for the member voxels.
int segment(const Image4DType::Pointer fmri_img, 
            const Image3DType::Pointer label_img,
            std::list< SectionType >& voxels)
{
    SectionType section; //will hold the label and iterator
    PixelIterator3D labelmap_it( label_img, label_img->GetRequestedRegion() );
    Image4DType::PointType phys_fmri; //stores a 4D point
    Image3DType::PointType phys_3D;   //stores a 3D point
    Image3DType::IndexType labelmap_index; //the index matching the fmri physical index
    PixelIterator4D fmri_it( fmri_img, fmri_img->GetRequestedRegion() );
    PixelIterator4D time_it( fmri_img, fmri_img->GetRequestedRegion() );
  
    //move in the slowest direction, and start at the beginning
    fmri_it.SetDirection(0);
    fmri_it.SetIndex(fmri_img->GetRequestedRegion().GetIndex());
   
    //timeit skips the entire 3D image, pointing at the first
    //voxel in the 3D image at for every time point, thus by
    //iterating you are pointing at the first point in the second
    //time step. The place we want to stop at
    time_it.SetDirection(3);
    time_it.SetIndex(fmri_img->GetRequestedRegion().GetIndex());
    ++time_it;


    //only iterate through the first time
    while(fmri_it != time_it) {
        while(!fmri_it.IsAtEndOfLine()) {
            //get the 4D point in the fmri image and then cut it to the first 3
            //Dimensions
            fmri_img->TransformIndexToPhysicalPoint( fmri_it.GetIndex(), phys_fmri);
            phys_3D[0] = phys_fmri[0];
            phys_3D[1] = phys_fmri[1];
            phys_3D[2] = phys_fmri[2];

#ifdef DEBUG
            printf("Pixel at: %f %f %f", phys_3D[0], phys_3D[1], phys_3D[2]);
#endif //DEBUG
            //check to see if that phsyical point in the fmri image is in the
            //label image
            if(label_img->TransformPhysicalPointToIndex(phys_3D, labelmap_index)) {
#ifdef DEBUG
                fprintf(stderr, "%li %li %li -> %li %li %li : %li\n", fmri_it.GetIndex()[0],
                        fmri_it.GetIndex()[1], fmri_it.GetIndex()[2], labelmap_index[0],
                        labelmap_index[1], labelmap_index[2], labelmap_it.Get());
#endif //DEBUG

                labelmap_it.SetIndex(labelmap_index);
                if( labelmap_it.Get() != 0) {
                    section.label = labelmap_it.Get();
                    section.point = fmri_it;
                    section.point.SetDirection(3); //step forward in time
                    voxels.push_front(section);
                }
            }
            ++fmri_it;
        }
        fmri_it.NextLine();
    }

    fprintf(stderr, "Finished loading input, sorting %lu nodes\n", voxels.size());
    voxels.sort(compare_lt);
    std::list<SectionType>::iterator it = voxels.begin();
    int prev_label = 0;
    int label_count = 0;
    while(it != voxels.end()) {
//        fprintf(stderr, "label: %d, prev_label: %i\n", it->label, prev_label);
        if(it->label != prev_label) {
            prev_label = it->label;
            label_count++;
            fprintf(stderr, "CHANGE %d\n", prev_label);
        }
        it++;
    }
    return label_count;
};

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
    fmri_img->CopyInformation(reader[0]->GetOutput());
    fmri_img->Update();

    delete[] reader;
    return fmri_img;
};

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

