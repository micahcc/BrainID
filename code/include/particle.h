#ifndef PARTICLE_H
#define PARTICLE_H

#include <itkNormalVariateGenerator.h>

template <typename State, typename U>
class pfilter {
public:
    pfilter();
    State sample();
    void update();

private:
    unsigned int num_particles;
    State* particles;
    State* weights;
    State* old_particles;
    double dt;
    double observe;
    double stim;
    double var;
    weighted_sample();
};

template <typename State, typename U>
pfilter::pfilter(int num_particles, State init)
{
    this->num_particles = num_particles;
    this->particles = new State(num_particles);
    this->old_particles = new State(num_particles);
    this->weights = new State(num_particles);
    //assign some sort of RV into the particles based
    //on init
}

//template <typename State, typename U>
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

template <typename State, typename U>
State pfilter::sample()
{
    State out;
    //sum up particles and return
    return out;
}

template <typename State, typename U>
int pfilter::update(U* caller, double delta_t)
{
    double norm_mean= 0;
    double norm_sig = 0;
    Grad_t curr_grad;

    itk::Statistics::NormalVariateGenerator rv;
    rv.initialize(rand());

    State::Iterator it;

    for(int i=0 ; i<num_particles ; i++) {

        //draw the new X_t from N(X_{t-1}+f(X_{t-1})dt, var*delta_t)
        //or x_t,i ~ p(x_t | u_t, x_t-1,i)
        particles[i] = caller->step(particles[i]);
        it = particles[i].Begin();
        while( it != particles[i].End() ) {
            *it = rv.variate()*sqrt(var*delta_t) + *it;
        }

        //draw weight from Normal, with mean equal to the difference
        //between the actual observed value and the calculated one
        //more generically, use the error as the mean and use
        //some sigma for the std. dev.
        weights[i] = caller->error(particles[i]);
        weights[i] = rv.variate()*sqrt(var) + weights[i];
    }

    double weight_sum;

    //sum up the inverse of the weights
    for(int i = 0 ; i<num_particles ; i++) {
        weights[i] = 1./weights[i];
        weight_sum += weights[i];
    }

    //this is just to keep from re-aquiring memory. The actual memory
    //for old_particles (which has just been moved to parcticles) will
    //be overwritten
    State* tmp = particles;
    particles = old_particles;
    old_particles = tmp;

    //need to copy random particles from old_particles which were
    //just updated to particles
    for(int i = 0 ; i<num_particles ; i++) {
        particles[i] = old_particles[weighted_sample(weight_sum)];
    }
}

template <typename State, typename U>
int weighted_sample(double max)
{
    double ran = (((double) rand())/RAND_MAX) * max;
    double sum = 0;
    for(int i=0 ; i<num_particles ; i++) {
        sum += weights[i];
        if(ran < sum) {
            return i;
        }
    }
    fprintf(stderr, "Hmm I don't think it should get here, at least it\n")
    fprintf(stderr, "should be very rare\n");
    return i;
}

#endif
