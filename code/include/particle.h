#ifndef PARTICLE_H
#define PARTICLE_H

#include "bold.h"

class pfilter {
public:
    pfilter();
    State_t  sample();

private:
    unsigned int num_particles;
    State_t* particles;
};

#endif
