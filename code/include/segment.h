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
Image4DType::Pointer getspline(const Image4DType::Pointer fmri_img,
            const std::vector<unsigned int>& knots);

Image3DType::Pointer get_average(const Image4DType::Pointer fmri_img);
Image3DType::Pointer get_variance(const Image4DType::Pointer fmri_img);
>>>>>>> 23c4099152aa6e288a616de086715af8ca74a830
Image3DType::Pointer extract(Image4DType::Pointer input, size_t index);

std::list<LabelType> getlabels(Label3DType::Pointer labelmap);

#endif
