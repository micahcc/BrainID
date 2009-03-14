#include "particle.h"
#include <cmath>
#include <assert.h>

pfilter::pfilter(int n_particles, int state_l, 
            void (*pxt)(const double*, void*, double[][2]), 
            void (*pyt)(const double*, void*, double[2]),
            void* extras = NULL)
{
    srand(time(NULL));
    num_particles = n_particles;
    state_len = state_l;

    particles = new double*[num_particles];
    for(uint32_t i=0 ; i<num_particles ; i++) {
        particles[i] = new double[state_len];
    }
    
    old_particles = new double*[num_particles];
    for(uint32_t i=0 ; i<num_particles ; i++) {
        old_particles[i] = new double[state_len];
    }

    this->weights = new double[num_particles];

    prxt_xtm1 = pxt;
    pryt_xtm1 = pyt;
    extra_data = extras;
    
    //assign some sort of RV into the particles based
    //on init
}

pfilter::~pfilter()
{
    for(uint32_t i=0 ; i<num_particles ; i++) {
        delete[] particles[i];
    }
    delete[] particles;
    
    for(uint32_t i=0 ; i<num_particles ; i++) {
        delete[] old_particles[i];
    }
    delete[] old_particles;

    delete[] weights;
}

//pfilter::pfilter(int num_particles, State::Pointer init)
//{
//    this->num_particles = num_particles;
//    this->particles = new State(num_particles);
//    //assign some sort of RV into the particles based
//
//    State::Iterator it = init->Begin();
//    while(it != init->End()) {
//        *it
//    }
//    //on init
//}

void pfilter::sample(double* dest)
{
    assert(dest);
    memcpy(dest, particles[(int)(rand()/(RAND_MAX+.1))*num_particles], state_len);
}

void pfilter::print()
{
    for(uint32_t i=0 ; i<num_particles ; i++){
        printf("( ");
        for(uint32_t j=0 ; j< state_len ; j++) {
            printf("%5.4f ", particles[i][j]);
        }
        printf("), ");
    }
}

int pfilter::update(void* extras=NULL)
{
    double dist_x[state_len][2];
    double dist_y[2]; //0 - mean, 1 - sigma^2
    double weight_sum;
    
    void* extra_local;

    int return_value = 0;
    
    if(extras) {
        extra_local = extras;
    } else if(extra_data) {
        extra_local = extra_data;
    } else {
        return_value += 1;
    }

    typedef itk::Statistics::NormalVariateGenerator Normal;
    Normal::Pointer rv = Normal::New();
    
    rv->Initialize(rand());

    for(uint32_t i=0 ; i<num_particles ; i++) {

        //draw the new X_t from N(X_{t-1}+f(X_{t-1})dt, var*delta_t)
        //or x_t,i ~ p(x_t | u_t, x_t-1,i)
        //particles[i] = caller->step(particles[i]);
        prxt_xtm1(particles[i], extra_local, dist_x);
        for(uint32_t j=0 ; j<state_len ; j++) {
            //de-normalize rv
            particles[i][j] = rv->GetVariate()*sqrt(dist_x[j][1]) + 
                        dist_x[j][0];
        }

        //draw weight from Normal, with mean equal to the difference
        //between the actual observed value and the calculated one
        //more generically, use the error as the mean and use
        //some sigma for the std. dev.
        pryt_xtm1(particles[i], extra_local, dist_y);
        weights[i] = rv->GetVariate()*sqrt(dist_y[1]) + dist_y[0];
    }

    //sum up the inverse of the weights
    for(uint32_t i = 0 ; i<num_particles ; i++) {
        weights[i] = 1./weights[i];
        weight_sum += weights[i];
    }

    //this is just to keep from re-aquiring memory. The actual memory
    //for old_particles (which has just been moved to parcticles) will
    //be overwritten
    double** tmp = particles;
    particles = old_particles;
    old_particles = tmp;

    //need to copy random particles from old_particles which were
    //just updated to particles
    for(uint32_t i = 0 ; i<num_particles ; i++) {
        memcpy(particles[i], old_particles[weighted_sample(weight_sum)], state_len);
    }

    return return_value;
}

int pfilter::weighted_sample(double max)
{
    double ran = (((double) rand())/RAND_MAX) * max;
    double sum = 0;
    uint32_t i;
    for(i=0 ; i<num_particles ; i++) {
        sum += weights[i];
        if(ran < sum) {
            return i;
        }
    }
    fprintf(stderr, "Hmm I don't think it should get here, at least it\n");
    fprintf(stderr, "should be very rare\n");
    return i;
}
