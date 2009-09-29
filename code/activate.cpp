
#include "activate.h"

#include <cmath>
#include <fstream>
#include <gsl/gsl_spline.h>
//#include <string>

void Activate::initialize(char* input)
{
    timepoints.clear();

    std::ifstream ifs(input);
    if(!ifs.is_open()) {
        return;
    }
    
    func tmp;
    double prev = 0;
    double time, level;

    while(true){
        ifs >> time;
        ifs >> level;
        if(ifs.fail() || ifs.eof()) {
            break;
        }

        tmp.delta = level-prev;
        tmp.t0 = time;
        this->timepoints.push_back(tmp);

        prev = level;
    }
};

Activate::Activate() : B(50)
{
    timepoints.clear();
};

Activate::Activate(char* input) : B(50)
{
    initialize(input);
};

//A*tanh(B*(t-t0)) + C
double Activate::at(double t) 
{
    double level = 0;
    std::vector<func>::iterator it = timepoints.begin();
    while(it != timepoints.end()) {
        level += (it->delta/2.)*tanh(B*(t-it->t0)) + it->delta/2.;
        it++;
    }
    return level;
};


