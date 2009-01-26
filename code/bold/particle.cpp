#include <cstdio>
#include <cstdlib>
#include "particle.h"

#include "bold.h"

pfilter::pfilter()
{
    this->num_particles = 10;
    this->particles = new State_t(num_particles);
}

State_t pfilter::sample()
{
    State_t out;
    return out;
}
