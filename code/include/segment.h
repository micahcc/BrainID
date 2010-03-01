#ifndef SEGMENT_H
#define SEGMENT_H

#include "types.h"

//standard libraries
#include <vector>
#include <list>

//Each SectionType struct contains an integer label
//and an iterator that moves forward in time
struct SectionType{
    int label;
    PixelIterator4D point;
} ;

enum { AVERAGES, LOCALMIN, LOCALMAX };

Image4DType::Pointer pruneFMRI(const Image4DType::Pointer fmri_img,
            std::vector<Activation>& stim, double dt,
            unsigned int remove);

Image4DType::Pointer deSplineByStim(const Image4DType::Pointer fmri_img,
            unsigned int numknots, std::vector<Activation>& stim, double dt);

Image4DType::Pointer deSplineBlind(const Image4DType::Pointer fmri_img,
            unsigned int numknots);

Image4DType::Pointer splitByRegion(const Image4DType::Pointer fmri_img,
            const Label3DType::Pointer labelmap, int label);

Image4DType::Pointer read_dicom(std::string directory, double skip = 0);

double get_average(const Image4DType::Pointer fmri_img, 
        const Label3DType::Pointer labelmap);

Image3DType::Pointer extract(Image4DType::Pointer input, size_t index);

std::list<LabelType> getlabels(Label3DType::Pointer labelmap);

#endif
