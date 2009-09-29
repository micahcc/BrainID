#ifndef ACTIVATE_H
#define ACTIVATE_H

#include <vector>
//#include <string>

class Activate
{
public:
    Activate(char* input);
    Activate();
    void initialize(char* input);

    double at(double t);
    double getlast() {
        if(!timepoints.empty())
            return timepoints.back().t0;
        else
            return 0;
    }

private:

    struct func{
        double t0;
        double delta;
    };
    
    std::vector<func> timepoints;
    const double B;
};

#endif //ACTIVATE_H
