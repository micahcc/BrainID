#ifndef PARTICLEFILTERMOD_HPP
#define PARTICLEFILTERMOD_HPP

class ParticleFilterMod : public ParticleFilter<double>
{
public:
    void print_particles() {
        for (unsigned int i = 0; i < this->p_xtn_ytn.getSize(); i++) {
            outputVector(std::cerr, this->p_xtn_ytn.get(i));
        }
    }

    ParticleFilterMod(ParticleFilterModel<double>* model,
            indii::ml::aux::DiracMixturePdf& p_x0);

    virtual void filter(const double tnp1, const indii::ml::aux::vector& ytnp1);
};

void ParticleFilterMod::filter(const double tnp1, const aux::vector& ytnp1) {
    
};

ParticleFilterMod::ParticleFilterMod(ParticleFilterModel<double>* model,
            indii::ml::aux::DiracMixturePdf& p_x0) : 
            ParticleFilter<double>(model, p_x0) {

};

#endif
