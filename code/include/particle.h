#ifndef PARTICLE_H
#define PARTICLE_H

#include <itkNormalVariateGenerator.h>
#include <itkVector.h>

static const int DEFAULT_NUM = 5000;

template <unsigned int LEN>
class PBase {
public:
    virtual itk::Vector<double, LEN> step(itk::Vector<double, LEN>) = 0;
    virtual double error(itk::Vector<double, LEN>) = 0;
    virtual ~PBase() { } ;
};

template <typename U, unsigned int LEN>
class ParticleF {
public:
    typedef itk::Vector<double, LEN> State;
    ParticleF();
    ParticleF(int, State);
    State sample();
    void update(U* caller, double delta_t);
    void print();
private:
    unsigned int num_particles;
    State* particles;
    double* weights;
    State* old_particles;
    double dt;
    double observe;
    double stim;
    double var;
    int weighted_sample(double);
};

template <typename U, unsigned int LEN>
ParticleF<U, LEN>::ParticleF()
{
    this->num_particles = DEFAULT_NUM;
    this->particles = new State(num_particles);
    this->old_particles = new State(num_particles);
    this->weights = new double(num_particles);
    this->var = 10;
    
    //assign some sort of RV into the particles based
    //on init

    typename ParticleF<U, LEN>::State::Iterator it;
    for(int i = 0 ; i<num_particles ; i++) {
        it = particles[i].Begin();
        while( it != particles[i].End() ) {
            *it = rand();
        }
    }
}

template <typename U, unsigned int LEN>
ParticleF<U, LEN>::ParticleF(int num_particles, State init)
{
    this->num_particles = num_particles;
    this->particles = new State(num_particles);
    this->old_particles = new State(num_particles);
    this->weights = new double(num_particles);
    
    //assign some sort of RV into the particles based
    //on init
    
    typename ParticleF<U, LEN>::State::Iterator it;
    for(int i = 0 ; i<num_particles ; i++) {
        it = particles[i].Begin();
        while( it != particles[i].End() ) {
            *it = rand();
        }
    }
}

//template <typename U, unsigned int LEN>
//ParticleF::ParticleF(int num_particles, State::Pointer init)
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

template <typename U, unsigned int LEN>
typename ParticleF<U, LEN>::State ParticleF<U, LEN>::sample()
{
    return particles[(((double) rand())/(RAND_MAX+1))*num_particles];
}

template <typename U, unsigned int LEN>
void ParticleF<U, LEN>::print()
{
    for(int i=0 ; i<num_particles ; i++){
        printf("%5.4f ", particles[i]);
    }
}

template <typename U, unsigned int LEN>
void ParticleF<U, LEN>::update(U* caller, double delta_t)
{
    double norm_mean= 0;
    double norm_sig = 0;

    itk::Statistics::NormalVariateGenerator rv;
    rv.Initialize(rand());

    //State::Iterator it;
    typename ParticleF<U, LEN>::State::Iterator it;

    for(int i=0 ; i<num_particles ; i++) {

        //draw the new X_t from N(X_{t-1}+f(X_{t-1})dt, var*delta_t)
        //or x_t,i ~ p(x_t | u_t, x_t-1,i)
        particles[i] = caller->step(particles[i]);
        it = particles[i].Begin();
        while( it != particles[i].End() ) {
            *it = rv.GetVariate()*sqrt(var*delta_t) + *it;
        }

        //draw weight from Normal, with mean equal to the difference
        //between the actual observed value and the calculated one
        //more generically, use the error as the mean and use
        //some sigma for the std. dev.
        weights[i] = caller->error(particles[i]);
        weights[i] = rv.GetVariate()*sqrt(var) + weights[i];
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

//returns the index of a particle based on the list of
//weights. To accomplish this it gets a random number
//between 0 and max. Then it adds weights in order until
//the max is exceeded. Thus the weights are used to convert
//a Uniformly distributed RV into a distribution as defined by the particles
template <typename U, unsigned int LEN>
int ParticleF<U, LEN>::weighted_sample(double max)
{
    double ran = (((double) rand())/RAND_MAX) * max;
    double sum = 0;
    int i;
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

#endif
