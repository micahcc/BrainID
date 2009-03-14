#ifndef PARTICLE_H
#define PARTICLE_H

#include <itkNormalVariateGenerator.h>

class pfilter {
public:
    pfilter(int, int, void (*pxt)(const double*, void*, double[][2]), 
                void (*pyt)(const double*, void*, double[2]), void*);
    ~pfilter();
    void sample(double* dest);
    int update(void*);
    void print();
private:
    void* extra_data;
    void (*prxt_xtm1)(const double*, void*, double[][2]); //pr(x_t | x_t-1) 
    void (*pryt_xtm1)(const double*, void*, double[2]); //pr(y_k | x_t-1)

    unsigned int num_particles;
    unsigned int state_len;
    
    double** particles;
    double** old_particles;

    double* weights;
    int weighted_sample(double max);
};


#endif
